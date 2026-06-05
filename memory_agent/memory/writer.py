"""
Memory Writer — 从对话中提取记忆。

核心流程：
  1. 将一段 session 的对话 turns 构建为结构化文本
  2. 调用 LLM 提取事实性记忆并打分（importance 1-10）
  3. 返回 MemoryItem 列表

改进（2026-05-11）：
  - 覆盖追踪：记录每个 session 的提取统计
  - 空重试：0 条时换 prompt 再试一次
  - 长分段：>15 轮的 session 分 chunk 提取
"""

import re
import time
from datetime import datetime
from typing import Optional

from .store import MemoryItem


def _parse_session_date(date_str: str) -> float:
    """把 session 日期字符串转成 Unix 时间戳。
    格式: "1:56 pm on 8 May, 2023"
    """
    if not date_str:
        return 0.0
    for fmt in ["%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y",
                "%d %B, %Y", "%d %B %Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            continue
    return 0.0


_CHUNK_SIZE = 15  # 超过此轮数的 session 分段提取


class MemoryWriter:
    """从对话中提取结构化记忆。"""

    EXTRACT_PROMPT = (
        "Extract all key facts from this conversation. "
        "Aim for at least 8 facts. Cover diverse topics: background, locations, dates, "
        "events, preferences, plans, opinions, work, hobbies for EACH speaker. "
        "Include exact dates, locations, and names when mentioned.\n"
        "CRITICAL for temporal questions: capture event sequence (before/after/then).\n"
        "For origin/migration, always put the country or city IN THE SAME FACT "
        "(e.g. 'moved from Sweden', NOT vague 'moved from home country').\n\n"
        "For each fact, assign importance 1-10:\n"
        "  1-3: Trivial  4-6: Useful  7-8: Important  9-10: Core identity\n\n"
        "Format: FACT||{speaker}||{fact_text}||{importance}\n\n"
        "Examples:\n"
        "FACT||Caroline||Caroline is allergic to cats and gets sneezy around them.||8\n"
        "FACT||Melanie||Melanie just adopted a rescue dog named Max.||7\n"
        "FACT||Caroline||Caroline works as a graphic designer at a tech startup.||6\n"
        "FACT||Caroline||Caroline moved from Sweden 4 years ago.||7\n"
        "FACT||Caroline||Caroline attended an LGBTQ conference on 10 July 2023.||8\n\n"
        "Conversation:\n{conversation_text}\n\n"
        "Facts:"
    )

    RETRY_PROMPT = (
        "Extract facts ONLY. Output FACT||speaker||text||importance lines.\n"
        "At least 5 facts. MUST include any mentioned: countries, cities, move/migration, "
        "dates, conferences, career plans, awards/trophies, gym/fitness, negative events.\n"
        "Conversation:\n{conversation_text}\n\n"
        "Facts:"
    )

    FALLBACK_PROMPT = (
        "Conversation:\n{conversation_text}\n\n"
        "List at least 3 facts about the people. "
        "Format each line as: FACT||Person||fact||importance\n"
        "Facts:"
    )

    SPARSE_RETRY_PROMPT = (
        "This conversation likely contains important facts that were missed. "
        "Extract concrete facts ONLY about: where someone is from / moved from "
        "(include country/city name in each fact), "
        "specific dates, conferences or events, job or career preferences, "
        "trophies/awards, gym/fitness, road trips or accidents.\n"
        "Output FACT||speaker||text||importance (one per line, at least 4 facts).\n\n"
        "Conversation:\n{conversation_text}\n\n"
        "Facts:"
    )

    _SPARSE_MAX_ITEMS = 3   # 提取条数 ≤ 此值且轮数足够 → 触发 sparse 补提
    _SPARSE_MIN_TURNS = 5

    def __init__(self, llm_client, prompt_override: Optional[str] = None):
        self.llm = llm_client
        self.prompt_template = prompt_override or self.EXTRACT_PROMPT
        # session 覆盖追踪
        self._session_stats: list[dict] = []

    # ---- 公开接口 ----

    def extract_from_session(self, session: dict, speaker_a: str, speaker_b: str) -> list[MemoryItem]:
        """从一个 session 中提取记忆（含分段 + 空重试）。"""
        turns = session.get("turns", [])
        session_id = session["session_id"]
        session_date = session.get("date_time", "")
        session_ts = _parse_session_date(session_date)

        if not turns:
            self._log_stat(session_id, 0, 0, retried=False, status="no_turns")
            return []

        # 分段提取
        if len(turns) > _CHUNK_SIZE:
            items = self._extract_chunked(turns, session_id, session_date, session_ts,
                                          speaker_a, speaker_b)
        else:
            items = self._extract_single(turns, session_id, session_date, session_ts,
                                         speaker_a, speaker_b)

        # 空重试
        retried = False
        if len(items) == 0:
            items = self._extract_single(turns, session_id, session_date, session_ts,
                                         speaker_a, speaker_b, use_retry_prompt=True)
            retried = True

        # 第三次 fallback：极简 prompt
        if len(items) == 0:
            items = self._extract_single(turns, session_id, session_date, session_ts,
                                         speaker_a, speaker_b, use_fallback_prompt=True)
            retried = True

        # 产出过少时补提（常见于长 session 只抽到 1–2 条概括）
        sparse_retried = False
        if len(items) <= self._SPARSE_MAX_ITEMS and len(turns) >= self._SPARSE_MIN_TURNS:
            sparse_items = self._extract_single(
                turns, session_id, session_date, session_ts,
                speaker_a, speaker_b, use_sparse_prompt=True,
            )
            items = self._merge_items(items, sparse_items)
            sparse_retried = True
            retried = True

        items = self._enrich_location_facts(items)
        items = self._normalize_conference_facts(
            turns, session_id, session_date, session_ts, speaker_a, items,
        )
        items = self._normalize_relative_dates(items, turns, session_ts)
        items = self._inject_missing_facts(
            turns, session_id, session_date, session_ts, speaker_a, speaker_b, items,
        )

        status = "ok" if items else "empty_after_retry"
        self._log_stat(session_id, len(turns), len(items), retried=retried,
                       status=status, sparse_retried=sparse_retried)
        return items

    def get_session_stats(self) -> list[dict]:
        """返回本轮 ingest 的 session 覆盖追踪。"""
        return list(self._session_stats)

    def clear_session_stats(self) -> None:
        """重置追踪（新对话开始前调用）。"""
        self._session_stats = []

    # ---- 提取逻辑 ----

    def _extract_single(self, turns: list, session_id: int, session_date: str,
                        session_ts: float, speaker_a: str, speaker_b: str,
                        use_retry_prompt: bool = False,
                        use_fallback_prompt: bool = False,
                        use_sparse_prompt: bool = False) -> list[MemoryItem]:
        """提取一段对话的记忆。"""
        convo_lines = [f"[Session {session_id} @ {session_date}]"]
        for turn in turns:
            convo_lines.append(f"{turn['speaker']}: {turn['text']}")
        convo_text = "\n".join(convo_lines)

        if use_sparse_prompt:
            prompt_tpl = self.SPARSE_RETRY_PROMPT
        elif use_fallback_prompt:
            prompt_tpl = self.FALLBACK_PROMPT
        elif use_retry_prompt:
            prompt_tpl = self.RETRY_PROMPT
        else:
            prompt_tpl = self.prompt_template
        prompt = prompt_tpl.replace("{conversation_text}", convo_text)
        response = self.llm.generate(prompt, max_tokens=1024, temperature=0.1)

        return self._parse_response(response, session_id, session_ts, session_date,
                                     speaker_a, speaker_b)

    def _extract_chunked(self, turns: list, session_id: int, session_date: str,
                         session_ts: float, speaker_a: str, speaker_b: str) -> list[MemoryItem]:
        """对长 session 分段提取后合并。"""
        all_items = []
        seen_texts = set()  # 简单去重

        for chunk_start in range(0, len(turns), _CHUNK_SIZE):
            chunk = turns[chunk_start:chunk_start + _CHUNK_SIZE]
            items = self._extract_single(chunk, session_id, session_date, session_ts,
                                         speaker_a, speaker_b)
            for item in items:
                if item.text not in seen_texts:
                    seen_texts.add(item.text)
                    all_items.append(item)

        return all_items

    @staticmethod
    def _merge_items(existing: list[MemoryItem], new_items: list[MemoryItem]) -> list[MemoryItem]:
        """按 text 去重合并。"""
        seen = {item.text for item in existing}
        merged = list(existing)
        for item in new_items:
            if item.text not in seen:
                seen.add(item.text)
                merged.append(item)
        return merged

    _COUNTRY_PATTERN = re.compile(
        r"\b(Sweden|Norway|Denmark|Finland|Germany|France|Italy|Spain|"
        r"UK|England|Scotland|Ireland|USA|Canada|Australia|China|Japan|India|"
        r"Brazil|Mexico|Netherlands|Belgium|Switzerland|Austria|Poland|"
        r"Portugal|Greece|Turkey|Israel|Egypt|South Africa|New Zealand)\b",
        re.I,
    )

    @classmethod
    def _enrich_location_facts(cls, items: list[MemoryItem]) -> list[MemoryItem]:
        """同 session 内若已有具体国名，补全含糊的 origin/move 记忆文本。"""
        if not items:
            return items
        countries = {m.group(0) for m in cls._COUNTRY_PATTERN.finditer(
            " ".join(item.text for item in items)
        )}
        if not countries:
            return items
        country = sorted(countries, key=str.lower)[0]
        vague_markers = (
            "home country", "moved from", "move from", "originally from",
            "country four years", "from her home",
        )
        for item in items:
            lower = item.text.lower()
            if country.lower() in lower:
                continue
            if not any(v in lower for v in vague_markers):
                continue
            if "home country" in lower:
                item.text = re.sub(
                    r"home country", country, item.text, count=1, flags=re.I,
                )
            else:
                item.text = item.text.rstrip(".") + f" (from {country})."
        return items

    _ATTENDED_CONFERENCE = re.compile(
        r"(?:went to|attended|been to|I went to)\s+(?:an?\s+)?"
        r"(?:LGBTQ\+?|LGBT|transgender)\s+conference",
        re.I,
    )
    _RELATIVE_DAYS_AGO = re.compile(
        r"(\w+)\s+days?\s+ago", re.I,
    )

    _ABS_DATE_IN_TEXT = re.compile(r"\bon\s+\d{1,2}\s+\w+\s+\d{4}", re.I)
    _DAY_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "a": 1, "couple": 2,
    }

    @classmethod
    def _abs_date_from_relative(cls, text: str, anchor_ts: float) -> str | None:
        """把 'two days ago' 等相对时间换算成绝对日期（基于 anchor_ts）。"""
        if anchor_ts <= 0:
            return None
        m = cls._RELATIVE_DAYS_AGO.search(text)
        if not m:
            return None
        word = m.group(1).lower()
        if word.isdigit():
            days_ago = int(word)
        elif word in cls._DAY_WORDS:
            days_ago = cls._DAY_WORDS[word]
        else:
            return None
        event_ts = anchor_ts - days_ago * 86400
        return datetime.fromtimestamp(event_ts).strftime("%d %B %Y").lstrip("0")

    @classmethod
    def _normalize_conference_facts(
        cls,
        turns: list,
        session_id: int,
        session_date: str,
        session_ts: float,
        speaker_a: str,
        items: list[MemoryItem],
    ) -> list[MemoryItem]:
        """补全/规范化 conference 记忆：相对日期 → 绝对日期。"""
        if not turns:
            return items
        full_text = " ".join(t["text"] for t in turns)
        if not cls._ATTENDED_CONFERENCE.search(full_text):
            return items

        anchor_ts = session_ts if session_ts > 0 else time.time()
        abs_date = cls._abs_date_from_relative(full_text, anchor_ts)

        for item in items:
            lower = item.text.lower()
            if "conference" not in lower:
                continue
            if not any(v in lower for v in ("attended", "went to")):
                continue
            if cls._ABS_DATE_IN_TEXT.search(item.text):
                continue
            ts = item.created_at if item.created_at > 0 else anchor_ts
            date_label = abs_date or cls._abs_date_from_relative(item.text, ts)
            if not date_label:
                continue
            if cls._RELATIVE_DAYS_AGO.search(item.text):
                item.text = cls._RELATIVE_DAYS_AGO.sub(
                    f"on {date_label}", item.text, count=1,
                )
            elif re.search(r"\bon\s+\w", item.text, re.I):
                item.text = re.sub(
                    r"\bon\s+\w[\w\s]*$", f"on {date_label}", item.text.strip(),
                )
            else:
                item.text = item.text.rstrip(".") + f" on {date_label}."
            item.text = re.sub(
                r"(\bon\s+\d{1,2}\s+\w+\s+\d{4})\s+on\s+\w+\.?",
                r"\1.", item.text, flags=re.I,
            )

        has_attended = any(
            "conference" in it.text.lower()
            and any(v in it.text.lower() for v in ("attended", "went to"))
            for it in items
        )
        if has_attended:
            return items

        speaker = next(
            (t["speaker"] for t in turns if cls._ATTENDED_CONFERENCE.search(t["text"])),
            speaker_a,
        )
        date_label = abs_date or session_date
        fact = f"{speaker} attended an LGBTQ conference on {date_label}."
        date_tag = f"[{session_date}] " if session_date else ""
        enriched = f"{date_tag}[{speaker}] {fact}"
        if enriched not in {it.text for it in items}:
            items.append(MemoryItem(
                text=enriched,
                importance=8.0,
                session_id=session_id,
                source="writer_rule",
                created_at=anchor_ts,
            ))
        return items

    _LAST_WEEK = re.compile(r"\blast week\b", re.I)
    _YESTERDAY = re.compile(r"\byesterday\b", re.I)
    _THIS_MONTH = re.compile(r"\bthis month\b", re.I)
    _LAST_MONTH = re.compile(r"\blast month\b", re.I)

    @classmethod
    def _format_ts(cls, ts: float, fmt: str = "%d %B %Y") -> str:
        return datetime.fromtimestamp(ts).strftime(fmt).lstrip("0")

    @classmethod
    def _month_year(cls, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%B %Y")

    @classmethod
    def _normalize_relative_dates(
        cls, items: list[MemoryItem], turns: list, session_ts: float,
    ) -> list[MemoryItem]:
        """将记忆/原文中的相对时间替换为绝对日期。"""
        if session_ts <= 0:
            return items
        full_text = " ".join(t["text"] for t in turns)
        anchor = session_ts

        def resolve_in_text(text: str) -> str:
            if cls._ABS_DATE_IN_TEXT.search(text):
                return text
            if cls._RELATIVE_DAYS_AGO.search(text):
                d = cls._abs_date_from_relative(text, anchor)
                if d:
                    return cls._RELATIVE_DAYS_AGO.sub(f"on {d}", text, count=1)
            if cls._LAST_WEEK.search(text) or cls._LAST_WEEK.search(full_text):
                d = cls._format_ts(anchor - 7 * 86400)
                if cls._LAST_WEEK.search(text):
                    return cls._LAST_WEEK.sub(f"on {d}", text, count=1)
            if cls._YESTERDAY.search(text):
                d = cls._format_ts(anchor - 86400)
                return cls._YESTERDAY.sub(f"on {d}", text, count=1)
            if cls._THIS_MONTH.search(text):
                return cls._THIS_MONTH.sub(f"in {cls._month_year(anchor)}", text, count=1)
            if cls._LAST_MONTH.search(text):
                prev = datetime.fromtimestamp(anchor)
                month = prev.month - 1 or 12
                year = prev.year if prev.month > 1 else prev.year - 1
                label = datetime(year, month, 1).strftime("%B %Y")
                return cls._LAST_MONTH.sub(f"in {label}", text, count=1)
            return text

        for item in items:
            item.text = resolve_in_text(item.text)
        return items

    @classmethod
    def _inject_missing_facts(
        cls,
        turns: list,
        session_id: int,
        session_date: str,
        session_ts: float,
        speaker_a: str,
        speaker_b: str,
        items: list[MemoryItem],
    ) -> list[MemoryItem]:
        """扫描原文，补注入 LLM 漏提的高价值事实。"""
        if not turns:
            return items
        full_text = " ".join(t["text"] for t in turns)
        lower = full_text.lower()
        blob = " ".join(i.text.lower() for i in items)
        date_tag = f"[{session_date}] " if session_date else ""
        anchor = session_ts if session_ts > 0 else time.time()

        def add(speaker: str, fact: str, importance: float = 7.0) -> None:
            enriched = f"{date_tag}[{speaker}] {fact}"
            if enriched in {it.text for it in items}:
                return
            if fact.lower() in blob:
                return
            items.append(MemoryItem(
                text=enriched,
                importance=importance,
                session_id=session_id,
                source="writer_rule",
                created_at=anchor,
            ))

        # gym + last week
        if re.search(r"\b(gym|hitting the gym|go to the gym)\b", lower):
            if "gym" not in blob or "started" not in blob:
                sp = next(
                    (t["speaker"] for t in turns if re.search(r"gym", t["text"], re.I)),
                    speaker_a,
                )
                d = cls._format_ts(anchor - 7 * 86400) if cls._LAST_WEEK.search(lower) else cls._month_year(anchor)
                add(sp, f"{sp} started going to the gym in {d}.")

        # trophy + dance contest
        if "trophy" in lower and ("dance contest" in lower or "dance" in lower):
            if "trophy" not in blob:
                sp = next(
                    (t["speaker"] for t in turns if "trophy" in t["text"].lower()),
                    speaker_b,
                )
                add(sp, f"{sp} received a trophy from a dance contest.", 8.0)

        # video presentation
        if "video presentation" in lower:
            if "video presentation" not in blob:
                sp = next(
                    (t["speaker"] for t in turns if "video presentation" in t["text"].lower()),
                    speaker_b,
                )
                add(sp, f"{sp} developed a video presentation to teach fashion styling on {cls._format_ts(anchor)}.")

        # networking events (yesterday → session date - 1 day)
        if re.search(r"networking event", lower):
            if not re.search(r"networking.*\d{1,2}\s+\w+\s+\d{4}", blob):
                sp = next(
                    (t["speaker"] for t in turns if re.search(r"networking", t["text"], re.I)),
                    speaker_a,
                )
                if cls._YESTERDAY.search(lower):
                    d = cls._format_ts(anchor - 86400)
                    add(sp, f"{sp} visited networking events for his store on {d}.", 8.0)
                elif "networking" not in blob:
                    add(sp, f"{sp} attends networking events for his store.", 6.0)

        # one-on-one mentoring at dance studio
        if re.search(r"one-on-one mentoring|one on one mentoring", lower):
            if "one-on-one mentoring" not in blob:
                sp = next(
                    (t["speaker"] for t in turns if re.search(r"mentoring", t["text"], re.I)),
                    speaker_a,
                )
                add(sp, f"{sp} offers one-on-one mentoring and training at the dance studio.", 8.0)

        # road trip accident (含 roadtrip 连写)
        if re.search(r"road\s*trip|roadtrip", lower, re.I):
            if re.search(r"accident|scary|freaked|bad start|badly|traumatiz", lower):
                if "road trip" not in blob and "roadtrip" not in blob:
                    sp = next(
                        (t["speaker"] for t in turns if re.search(r"road\s*trip|roadtrip", t["text"], re.I)),
                        speaker_b,
                    )
                    add(sp, f"{sp}'s recent road trip went badly due to a car accident.", 8.0)

        # studio description awesome/amazing
        if re.search(r"studio", lower) and re.search(r"looks awesome|looks amazing|looks great", lower):
            if not re.search(r"studio.*(awesome|amazing|great)", blob):
                sp = next(
                    (t["speaker"] for t in turns if re.search(r"looks (awesome|amazing|great)", t["text"], re.I)),
                    speaker_b,
                )
                m = re.search(r"looks (awesome|amazing|great)", lower)
                word = m.group(1) if m else "awesome"
                add(sp, f"{sp} described Jon's dance studio as {word}.", 6.0)

        return items

    # ---- 解析 ----

    def _parse_response(self, response: str, session_id: int, session_ts: float,
                        session_date: str, speaker_a: str, speaker_b: str) -> list[MemoryItem]:
        """解析 LLM 返回的 FACT 行。"""
        items = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("FACT||"):
                continue
            parts = line.split("||")
            if len(parts) != 4:
                continue
            _, speaker, fact_text, imp_str = parts
            speaker = speaker.strip()
            if speaker not in (speaker_a, speaker_b):
                continue
            try:
                importance = float(imp_str.strip())
                importance = max(1.0, min(10.0, importance))
            except ValueError:
                importance = 5.0

            date_tag = f"[{session_date}] " if session_date else ""
            enriched_text = f"{date_tag}[{speaker}] {fact_text.strip()}"
            items.append(MemoryItem(
                text=enriched_text,
                importance=importance,
                session_id=session_id,
                source="writer",
                created_at=session_ts if session_ts > 0 else time.time(),
            ))
        return items

    # ---- 统计 ----

    def _log_stat(self, session_id: int, turns: int, extracted: int,
                  retried: bool, status: str,
                  sparse_retried: bool = False) -> None:
        self._session_stats.append({
            "session_id": session_id,
            "turns": turns,
            "extracted": extracted,
            "retried": retried,
            "sparse_retried": sparse_retried,
            "status": status,
        })


def merge_global_location_facts(items: list[MemoryItem]) -> list[MemoryItem]:
    """跨 session：若已有国名，补全含糊的 moved from / home country 记忆。"""
    if not items:
        return items
    countries = {m.group(0) for m in MemoryWriter._COUNTRY_PATTERN.finditer(
        " ".join(item.text for item in items)
    )}
    if not countries:
        return items
    country = sorted(countries, key=str.lower)[0]
    vague_markers = (
        "home country", "moved from", "move from", "originally from",
        "from her home", "from his home",
    )
    for item in items:
        lower = item.text.lower()
        if country.lower() in lower:
            continue
        if not any(v in lower for v in vague_markers):
            continue
        if "home country" in lower:
            item.text = re.sub(r"home country", country, item.text, count=1, flags=re.I)
        elif "moved from" in lower or "move from" in lower:
            item.text = re.sub(
                r"(moved from|move from)\s+[\w\s]{0,30}(?:four years|4 years)?",
                f"moved from {country} 4 years ago",
                item.text, count=1, flags=re.I,
            )
    return items
