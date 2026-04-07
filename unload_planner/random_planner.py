"""Uniform random choice among active boxes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import RemovalPlanner


class RandomRemovalPlanner(RemovalPlanner):
    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def select_body_to_remove(self, active_body_indices: Sequence[int]) -> int:
        if not active_body_indices:
            raise ValueError("RandomRemovalPlanner: no active bodies to remove.")
        return int(self._rng.choice(list(active_body_indices)))
