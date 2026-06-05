"""
Memory Store — 记忆存储与索引。

核心数据结构 MemoryItem 表示一条派生记忆，
MemoryStore 管理所有 MemoryItem 并提供向量检索能力。

设计原则：
  - 明确区分"原始对话日志"（不在本模块管理）和"派生的记忆单元"（本模块管理）
  - 每条记忆保留完整元数据链路，支持事后溯源
  - 向量索引（FAISS）和标量元数据索引分离，便于消融时灵活切换检索策略
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 记忆单元定义
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """一条结构化记忆。"""
    text: str                                         # 记忆正文（自然语言）
    importance: float = 5.0                           # 重要性 1-10
    session_id: int = 0                               # 来源 session
    source: str = "writer"                            # "writer" | "reflection" | "summary"
    level: str = "low"                                # "low" 具体事实 | "high" 主题摘要
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)  # 可用于覆盖为对话的真实时间戳
    access_count: int = 0                             # 被检索命中次数（用于遗忘曲线复习）
    embedding: Optional[np.ndarray] = None            # 向量（惰性赋值）

    def __repr__(self) -> str:
        return (f"MemoryItem(id={self.memory_id[:8]}, "
                f"level={self.level}, imp={self.importance:.1f}, session={self.session_id}, "
                f"'{self.text[:40]}...')")


# ---------------------------------------------------------------------------
# 记忆存储与索引
# ---------------------------------------------------------------------------

class MemoryStore:
    """管理 MemoryItem 的存储、向量索引和检索。"""

    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
        self._items: list[MemoryItem] = []            # 有序列表，索引即 item_id
        self._id_to_idx: dict[str, int] = {}          # memory_id → list index

        # FAISS 索引：余弦相似度（Inner Product on normalized vectors）
        self._index: Optional["faiss.Index"] = None
        self._index_dirty = False                     # 是否有新增未重建索引

    # ---- 写操作 ----

    def add_item(self, item: MemoryItem) -> str:
        """添加一条记忆。如果已附带 embedding 则直接加入索引。"""
        idx = len(self._items)
        self._items.append(item)
        self._id_to_idx[item.memory_id] = idx
        if item.embedding is not None:
            self._lazy_index()
            self._index.add(np.expand_dims(item.embedding, 0).astype(np.float32))
        else:
            self._index_dirty = True
        return item.memory_id

    def add_items(self, items: list[MemoryItem]) -> list[str]:
        """批量添加记忆。"""
        return [self.add_item(it) for it in items]

    def update_access(self, memory_id: str) -> None:
        """更新记忆的被访问次数（复习计数）。"""
        idx = self._id_to_idx.get(memory_id)
        if idx is not None:
            self._items[idx].access_count += 1

    # ---- 读操作 ----

    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        idx = self._id_to_idx.get(memory_id)
        return self._items[idx] if idx is not None else None

    def get_all(self) -> list[MemoryItem]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    # ---- 向量检索 ----

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[MemoryItem]:
        """返回与 query_vec 最相似的 top_k 条记忆（余弦相似度）。"""
        if len(self._items) == 0:
            return []
        k = min(top_k, len(self._items))
        self._rebuild_index_if_dirty()
        # 确保 query_vec 是 2D float32
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
        scores, indices = self._index.search(query_vec.astype(np.float32), k)
        return [self._items[i] for i in indices[0]]

    def search_with_scores(self, query_vec: np.ndarray, top_k: int = 10
                           ) -> list[tuple[MemoryItem, float]]:
        """返回 (记忆, 余弦相似度) 列表。"""
        if len(self._items) == 0:
            return []
        k = min(top_k, len(self._items))
        self._rebuild_index_if_dirty()
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
        scores, indices = self._index.search(query_vec.astype(np.float32), k)
        return [(self._items[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

    # ---- 内部方法 ----

    def _lazy_index(self):
        """延迟初始化 FAISS 索引。"""
        if self._index is not None:
            return
        import faiss
        self._index = faiss.IndexFlatIP(self.embed_dim)

    def _rebuild_index_if_dirty(self):
        """如果存在未索引的 items，重建整个索引。"""
        if not self._index_dirty:
            return
        self._lazy_index()
        vecs = []
        for item in self._items:
            if item.embedding is not None:
                vecs.append(item.embedding)
        if vecs:
            self._index = faiss.IndexFlatIP(self.embed_dim)
            self._index.add(np.array(vecs, dtype=np.float32))
        self._index_dirty = False

    def rebuild_index(self, embeddings: np.ndarray):
        """用外部提供的完整 embedding 矩阵重建索引。"""
        self._lazy_index()
        self._index.reset()
        self._index.add(embeddings.astype(np.float32))
        self._index_dirty = False
