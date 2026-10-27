"""Data schemas and types for the mini-infer Agent Memory Engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MemoryCategory(StrEnum):
    """Categorization for hierarchical agent memory items."""

    CORE = "core"  # Pinned user rules, hardware specs, never decayed
    PROJECT = "project"  # Codebase structure, APIs, conventions
    DECISION = "decision"  # Architectural decisions, trade-offs made
    EPISODIC = "episodic"  # Historical conversations, user questions, tool runs


@dataclass
class MemoryItem:
    """A discrete persistent memory unit stored in the Agent Memory Engine."""

    content: str
    category: MemoryCategory = MemoryCategory.PROJECT
    importance: float = 1.0  # Priority weight: 0.1 (trivial) to 2.0 (critical)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory item to dictionary for serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Construct memory item from dictionary record."""
        data_copy = dict(data)
        if isinstance(data_copy.get("category"), str):
            data_copy["category"] = MemoryCategory(data_copy["category"])
        return cls(**data_copy)

    def compute_decay(
        self, current_time: float | None = None, half_life_days: float = 14.0
    ) -> float:
        """Compute time-decay multiplier for relevance scoring.

        CORE category memories never decay. Other categories experience exponential decay
        based on time since last access, reinforced by frequent access count.
        """
        if self.category == MemoryCategory.CORE:
            return 1.0

        now = current_time if current_time is not None else time.time()
        elapsed_seconds = max(0.0, now - self.last_accessed)
        elapsed_days = elapsed_seconds / 86400.0

        # Frequency reinforcement: each access extends effective freshness
        decay_rate = 0.69315 / max(half_life_days, 1.0)  # ln(2) / half_life
        reinforcement = 1.0 + min(self.access_count * 0.1, 1.0)

        # Exponential decay: e^(-lambda * t) * reinforcement
        import math

        decay_factor = math.exp(-decay_rate * elapsed_days) * reinforcement
        return min(max(decay_factor, 0.05), 1.5)
