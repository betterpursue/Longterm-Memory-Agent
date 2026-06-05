"""
Memory Updater — 记忆更新 / 遗忘模块。

提供两种实现：
  - NoOpUpdater:   不做任何更新（用于"无遗忘"对照实验）
  - EbbinghausUpdater: 基于 Ebbinghaus 遗忘曲线的记忆动力学

Ebbinghaus 公式：R(t) = e^{-t / S}
  - t = 当前时间 - 记忆创建时间（秒）
  - S = 记忆强度，取决于初始重要性和复习次数
  - 每次检索命中（复习）时 S += ΔS
"""

import time
import math
from abc import ABC, abstractmethod
from typing import Optional

from .store import MemoryItem


class BaseUpdater(ABC):
    """更新/遗忘基类。"""

    @abstractmethod
    def get_retention(self, memory: MemoryItem) -> float:
        """返回记忆的保留率 [0, 1]。1=完全保留，0=完全遗忘。"""
        ...

    @abstractmethod
    def on_access(self, memory: MemoryItem) -> None:
        """记忆被检索命中时调用（复习）。"""
        ...

    def set_ref_time(self, ref_time: float) -> None:
        """设置遗忘曲线的参考时间（默认为 time.time()）。"""
        pass

    def adaptive_base_strength(self, span_seconds: float) -> None:
        """根据对话时间跨度自适应调整遗忘强度。"""
        pass

    def update_after_answer(self, memories: list[MemoryItem]) -> None:
        """在 answer() 结束后批量更新。"""
        for mem in memories:
            self.on_access(mem)


# ---------------------------------------------------------------------------
# NoOp（无遗忘）
# ---------------------------------------------------------------------------

class NoOpUpdater(BaseUpdater):
    """无遗忘：所有记忆永远 100% 保留。用于对照实验。"""

    def get_retention(self, memory: MemoryItem) -> float:
        return 1.0

    def on_access(self, memory: MemoryItem) -> None:
        pass


# ---------------------------------------------------------------------------
# Ebbinghaus 遗忘曲线
# ---------------------------------------------------------------------------

class EbbinghausUpdater(BaseUpdater):
    """基于 Ebbinghaus 遗忘曲线。

    R(t) = e^{-t / S}

    记忆强度 S 的初始化与更新：
      S₀ = base_strength × importance_multiplier
      importance_multiplier = 1 + (importance - 1) / 3   (importance 1-10)
      S += strength_increment   (每次复习)
    """

    def __init__(self, base_strength: float = 2592000.0,   # 默认强度 30 天（秒）
                 strength_increment: float = 864000.0,       # 每次复习增加 10 天强度
                 retention_threshold: float = 0.05):          # 软降权阈值
        """
        Args:
            base_strength: 基础记忆强度（秒）。importance=1 时的 S 值
            strength_increment: 每次复习增加的强度（秒）
            retention_threshold: 保留率低于此阈值时视为"已遗忘"

        LoCoMo 对话时间跨度为数月至一年，
        base_strength 需设到月级别，否则旧 session 的记忆全部遗忘。
        """
        self.base_strength = base_strength
        self.strength_increment = strength_increment
        self.retention_threshold = retention_threshold
        # 参考时间：默认为 time.time()，可通过 set_ref_time() 改为对话的最后 session 时间
        self._ref_time: float | None = None

    def _compute_strength(self, memory: MemoryItem) -> float:
        """计算记忆强度 S。"""
        # 重要性乘数：importance=1 → 1.0, importance=10 → 4.0
        imp_mult = 1.0 + (memory.importance - 1.0) / 3.0
        S = self.base_strength * imp_mult
        # 复习增加强度（每次复习让 S 增加固定值）
        S += memory.access_count * self.strength_increment
        return S

    def set_ref_time(self, ref_time: float) -> None:
        """设置参考时间（通常为对话的最后一个 session 时间戳）。
        遗忘曲线的"现在"将用这个时间而非 time.time()。
        """
        self._ref_time = ref_time

    def adaptive_base_strength(self, span_seconds: float) -> None:
        """根据对话时间跨度自适应设置 base_strength。
        span_seconds = 最晚 session - 最早 session 的秒数。
        base_strength 设为 span × 2，使最早的记忆保留率 ≈ 0.37。
        注意：调用者传入原始 span 即可，本方法内部 ×2。
        """
        self.base_strength = max(span_seconds * 2, 86400.0)

    def get_retention(self, memory: MemoryItem) -> float:
        """Ebbinghaus 保留率。"""
        now = self._ref_time if self._ref_time is not None else time.time()
        age = now - memory.created_at
        if age <= 0:
            return 1.0
        S = self._compute_strength(memory)
        R = math.exp(-age / S)
        return max(0.0, min(1.0, R))

    def on_access(self, memory: MemoryItem) -> None:
        """复习：增加 access_count。实际的强度更新在下次 get_retention 时计算。"""
        memory.access_count += 1

    def is_forgotten(self, memory: MemoryItem) -> bool:
        """是否已被遗忘（低于阈值）。"""
        return self.get_retention(memory) < self.retention_threshold
