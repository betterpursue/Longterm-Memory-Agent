"""
汇总全部 Judge 结果 — 总表、分题型表、延迟与 LLM 调用估算。

用法：
    python eval/summarize_all.py experiments/results/
    python eval/summarize_all.py experiments/results/*_judge.json
"""

import json
import sys
from pathlib import Path

CATEGORIES = ("single_hop", "temporal", "multi_hop", "open_domain")


def _load_judge(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _pred_path(judge_path: Path) -> Path | None:
    """从 judge JSON 推断原始 predictions 路径。"""
    with open(judge_path, encoding="utf-8") as f:
        data = json.load(f)
    p = data.get("predictions_file")
    if p:
        cand = Path(p)
        if not cand.is_absolute():
            cand = judge_path.parent / cand
        if cand.exists():
            return cand
    stem = judge_path.name.replace("_judge_v2", "").replace("_judge", "")
    if not stem.endswith(".json"):
        stem += ".json"
    cand = judge_path.parent / stem.replace("_judge.json", ".json")
    if cand.exists():
        return cand
    base = judge_path.stem.replace("_judge_v2", "").replace("_judge", "")
    cand = judge_path.parent / f"{base}.json"
    return cand if cand.exists() else None


def _estimate_llm_calls(preds: list) -> float:
    """从 trace 估算每题 LLM 调用次数（answer 至少 1 次；ingest 摊销到每题）。"""
    if not preds:
        return 0.0
    answer_calls = 0
    ingest_sessions = 0
    conv_ids = set()
    for p in preds:
        trace = p.get("trace") or {}
        retrieved = trace.get("retrieved")
        if retrieved is not None:
            answer_calls += 1
        qa_id = p.get("qa_id", "")
        if "_q" in qa_id:
            conv_ids.add(qa_id.rsplit("_q", 1)[0])
    # ingest 调用无法从单题 trace 精确统计，用 0 表示仅 answer 阶段
    return round(answer_calls / len(preds), 2)


def _exp_name(path: Path) -> str:
    name = path.stem.replace("_judge_v2", "").replace("_judge", "")
    return name


def summarize_files(paths: list[Path], out_md: Path | None = None) -> str:
    # 先构建原始行列表
    raw_rows = []
    for jp in sorted(paths):
        data = _load_judge(jp)
        if not data or "overall" not in data:
            continue
        overall = data["overall"]
        by_cat = data.get("by_category", {})
        pred_path = _pred_path(jp)
        preds = []
        if pred_path:
            with open(pred_path, encoding="utf-8") as f:
                preds = json.load(f)
        errors = sum(1 for p in preds if p.get("error"))
        raw_rows.append({
            "name": _exp_name(jp),
            "n": overall.get("n", 0),
            "score": overall.get("score", 0),
            "f1": overall.get("f1", 0),
            "em": overall.get("em", 0),
            "latency": overall.get("avg_latency_sec", 0),
            "by_cat": by_cat,
            "errors": errors,
            "llm_calls_per_q": _estimate_llm_calls(preds),
            "_stem": jp.stem,  # 保留文件名，用于去重排序
        })

    # 去重：同名的只保留最佳版本。排序优先级：v2 > 高 score > 多 n
    def _dedup_key(r: dict) -> tuple:
        stem = r.get("_stem", "")
        is_v2 = "v2" in stem
        return (is_v2, r["score"], r["n"])

    seen_names: set[str] = set()
    rows = []
    for r in sorted(raw_rows, key=_dedup_key, reverse=True):
        name = r["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        # 移除内部字段
        del r["_stem"]
        rows.append(r)

    lines = [
        "# 实验汇总",
        "",
        "## 总体",
        "",
        "| 实验 | 题数 | 得分 | F1 | EM | 平均延迟(s) | 错误 |",
        "|------|------|------|-----|-----|-------------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['score']:.3f} | {r['f1']:.3f} | "
            f"{r['em']:.3f} | {r['latency']:.3f} | {r['errors']} |"
        )

    lines.extend(["", "## 分题型得分", ""])
    header = "| 实验 | " + " | ".join(CATEGORIES) + " |"
    sep = "|------|" + "|".join(["------"] * len(CATEGORIES)) + "|"
    lines.extend([header, sep])
    for r in rows:
        cells = [r["name"]]
        for cat in CATEGORIES:
            d = r["by_cat"].get(cat, {})
            cells.append(f"{d.get('score', 0):.3f}" if d else "—")
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text, encoding="utf-8")
        print(f"已写入 {out_md}")
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_all.py <results_dir_or_judge_files...>")
        sys.exit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*_judge*.json")))
        elif p.exists():
            paths.append(p)

    paths = [p for p in paths if "_judge" in p.name and p.suffix == ".json"]
    if not paths:
        print("未找到 *_judge*.json 文件")
        sys.exit(1)

    out_md = paths[0].parent / "summary_table.md"
    text = summarize_files(paths, out_md)
    print(text)


if __name__ == "__main__":
    main()
