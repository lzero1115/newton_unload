"""Construct a removal planner by name."""

from __future__ import annotations

import numpy as np

from .base import RemovalPlanner, UnloadSimLike
from .height_planner import HeightRemovalPlanner
from .random_planner import RandomRemovalPlanner


def make_planner(name: str, rng: np.random.Generator, sim: UnloadSimLike) -> RemovalPlanner:
    if name == "random":
        return RandomRemovalPlanner(rng)
    if name == "height":
        return HeightRemovalPlanner(sim)
    raise ValueError(f"Unknown removal planner: {name!r}. Supported: 'random', 'height'.")
