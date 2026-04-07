"""Abstract base and typing helpers for unload removal policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence


class UnloadSimLike(Protocol):
    """Minimal surface for planners that read rigid body poses from Newton state."""

    state_0: Any


class RemovalPlanner(ABC):
    """Choose which global body index to remove next from ``active_body_indices``."""

    @abstractmethod
    def select_body_to_remove(self, active_body_indices: Sequence[int]) -> int:
        """Return one body index from ``active_body_indices`` (must be non-empty)."""
