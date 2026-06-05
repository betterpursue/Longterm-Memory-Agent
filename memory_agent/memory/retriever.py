"""
Memory Retriever — 检索策略模块。

提供三种检索策略，可通过开关自由切换（用于消融实验）：

  1. DenseRetriever    — 纯稠密检索（基线对照）
     只用 embedding 余弦相似度排序

  2. ThreeFactorRetriever — 三因子检索（探索方向 A）
     Score = α·Relevance + β·Recency + γ·Importance
     其中 Relevance 为归一化的余弦相似度

  3. ForgettingRetriever — 带遗忘的三因子检索（探索方向 A + B 联合）
     Score' = (α·Relevance + β·Recency + γ·Importance) × R(t)
     其中 R(t) 来自 MemoryUpdater 的遗忘保留率

用法：
    retriever = ThreeFactorRetriever(alpha=0.5, beta=0.2, gamma=0.3)
    results = retriever.retrieve(query_vec, memories, top_k=5)
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .store import MemoryItem
from .updater import BaseUpdater


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """检索策略基类。"""

    @abstractmethod
    def retrieve(self, query_vec: np.ndarray, memories: list[MemoryItem],
                 top_k: int = 10, updater: Optional[BaseUpdater] = None,
                 query_text: str = ""
                 ) -> list[tuple[MemoryItem, float]]:
        ...

    def _log_retrieval(self, memories: list[tuple[MemoryItem, float]],
                       scores_detail: dict = None):
        """提供一个钩子，子类可覆盖以记录检索详情。"""
        pass


# ---------------------------------------------------------------------------
# 策略 1：纯稠密检索
# ---------------------------------------------------------------------------

class DenseRetriever(BaseRetriever):
    """纯稠密检索：仅使用 embedding 余弦相似度。"""

    def retrieve(self, query_vec: np.ndarray, memories: list[MemoryItem],
                 top_k: int = 10, updater: Optional[BaseUpdater] = None,
                 query_text: str = ""
                 ) -> list[tuple[MemoryItem, float]]:
        if not memories:
            return []
        k = min(top_k, len(memories))

        # 计算 cosine 相似度
        scores = []
        for mem in memories:
            if mem.embedding is None:
                continue
            sim = float(query_vec @ mem.embedding)
            scores.append((mem, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ---------------------------------------------------------------------------
# 策略 2：三因子检索
# ---------------------------------------------------------------------------

class ThreeFactorRetriever(BaseRetriever):
    """三因子检索：Relevance × Recency × Importance。

    Score = α · Relevance + β · Recency_norm + γ · Importance_norm

    各因子归一化到 [0, 1] 区间后加权求和。
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.2, gamma: float = 0.3,
                 recency_halflife_hours: float = 72.0):
        """
        Args:
            alpha: Relevance 权重（语义相似度）
            beta:  Recency 权重（近因性）
            gamma: Importance 权重（重要性）
            recency_halflife_hours: 近因性半衰期（小时）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recency_halflife = recency_halflife_hours * 3600  # 转为秒

    def retrieve(self, query_vec: np.ndarray, memories: list[MemoryItem],
                 top_k: int = 10, updater: Optional[BaseUpdater] = None,
                 query_text: str = ""
                 ) -> list[tuple[MemoryItem, float]]:
        if not memories:
            return []
        k = min(top_k, len(memories))

        now = time.time()
        scored: list[tuple[MemoryItem, float, float, float, float]] = []

        for mem in memories:
            if mem.embedding is None:
                continue

            # 1. Relevance：余弦相似度
            relevance = float(query_vec @ mem.embedding)

            # 2. Recency：指数衰减
            age = now - mem.created_at
            recency = np.exp(-age / self.recency_halflife)

            # 3. Importance：原始分数归一化（已限制在 1-10）
            importance = mem.importance / 10.0

            total = self.alpha * relevance + self.beta * recency + self.gamma * importance
            scored.append((mem, total, relevance, recency, importance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(mem, score) for mem, score, *_ in scored[:k]]


# ---------------------------------------------------------------------------
# 策略 3：带遗忘的三因子检索
# ---------------------------------------------------------------------------

class ForgettingRetriever(BaseRetriever):
    """带遗忘衰减的三因子检索。

    Score' = (α · Relevance + β · Recency + γ · Importance) × R(t)

    R(t) 由 MemoryUpdater 提供，表示当前保留率 [0, 1]。
    如果没有提供 updater，退化 = ThreeFactorRetriever。
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.15, gamma: float = 0.15,
                 recency_halflife_hours: float = 2160.0):  # 90 天
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recency_halflife = recency_halflife_hours * 3600
        self._ref_time: float | None = None

    def set_ref_time(self, ref_time: float) -> None:
        """设置近因性的参考时间（应与遗忘曲线的参考时间一致）。"""
        self._ref_time = ref_time

    def retrieve(self, query_vec: np.ndarray, memories: list[MemoryItem],
                 top_k: int = 10, updater: Optional[BaseUpdater] = None,
                 query_text: str = ""
                 ) -> list[tuple[MemoryItem, float]]:
        if not memories:
            return []
        k = min(top_k, len(memories))

        now = self._ref_time if self._ref_time is not None else time.time()
        scored: list[tuple[MemoryItem, float]] = []

        for mem in memories:
            if mem.embedding is None:
                continue

            relevance = float(query_vec @ mem.embedding)
            age = now - mem.created_at
            recency = np.exp(-age / self.recency_halflife)
            importance = mem.importance / 10.0

            total = self.alpha * relevance + self.beta * recency + self.gamma * importance

            # 遗忘衰减
            if updater is not None:
                retention = updater.get_retention(mem)
                total *= retention

            scored.append((mem, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
