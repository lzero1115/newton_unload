"""Remove the active box whose body origin has the largest world-frame z (highest first)."""

from __future__ import annotations

from typing import Sequence

from .base import RemovalPlanner, UnloadSimLike


class HeightRemovalPlanner(RemovalPlanner):
    def __init__(self, sim: UnloadSimLike):
        self._sim = sim

    def select_body_to_remove(self, active_body_indices: Sequence[int]) -> int:
        if not active_body_indices:
            raise ValueError("HeightRemovalPlanner: no active bodies to remove.")
        bq = self._sim.state_0.body_q.numpy()
        best_i = int(active_body_indices[0])
        best_z = float(bq[best_i][2])
        for idx in active_body_indices:
            i = int(idx)
            z = float(bq[i][2])
            if z > best_z or (z == best_z and i < best_i):
                best_z = z
                best_i = i
        return best_i
