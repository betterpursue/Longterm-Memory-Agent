"""
Ingest 结果缓存 — 同一对话多次评测时跳过 Writer LLM 调用。

缓存 key = conversation_id + WRITER_CACHE_VERSION
默认目录：memory_agent/experiments/ingest_cache/

环境变量：
  INGEST_CACHE=1       启用缓存（默认开启）
  INGEST_CACHE=0       禁用
  WRITER_CACHE_VERSION  bump 后使旧缓存失效
"""

import json
import os
from pathlib import Path

import numpy as np

from memory.store import MemoryItem

_CACHE_DIR = Path(__file__).resolve().parent.parent / "experiments" / "ingest_cache"
WRITER_CACHE_VERSION = os.getenv("WRITER_CACHE_VERSION", "v3")
_CACHE_ENABLED = os.getenv("INGEST_CACHE", "1") == "1"


def _cache_path(conversation_id: str) -> Path:
    safe = conversation_id.replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{safe}_{WRITER_CACHE_VERSION}.json"


def load_cached_items(conversation_id: str) -> list[MemoryItem] | None:
    """命中则返回 MemoryItem 列表（含 embedding）；未命中返回 None。"""
    if not _CACHE_ENABLED or not conversation_id:
        return None
    path = _cache_path(conversation_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for row in data.get("items", []):
            emb = row.get("embedding")
            item = MemoryItem(
                text=row["text"],
                importance=row.get("importance", 5.0),
                session_id=row.get("session_id", 0),
                source=row.get("source", "writer"),
                memory_id=row.get("memory_id", ""),
                created_at=row.get("created_at", 0.0),
                access_count=row.get("access_count", 0),
                embedding=np.array(emb, dtype=np.float32) if emb else None,
            )
            items.append(item)
        return items if items else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_cached_items(conversation_id: str, items: list[MemoryItem]) -> None:
    """将 ingest 结果写入缓存。"""
    if not _CACHE_ENABLED or not conversation_id or not items:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        rows.append({
            "text": item.text,
            "importance": item.importance,
            "session_id": item.session_id,
            "source": item.source,
            "memory_id": item.memory_id,
            "created_at": item.created_at,
            "access_count": item.access_count,
            "embedding": item.embedding.tolist() if item.embedding is not None else None,
        })
    with open(_cache_path(conversation_id), "w", encoding="utf-8") as f:
        json.dump({"conversation_id": conversation_id, "items": rows}, f, ensure_ascii=False)
