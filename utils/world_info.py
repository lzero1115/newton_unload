"""
Per-world diagnostics, collision-pipeline sizing, and container wall geometry.

Reuse from unload_clean / repo scripts as needed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import warp as wp


def load_snapshot_metadata(npz_data: Any) -> dict[str, Any]:
    """
    Parse ``metadata_json`` from a settled-scene ``.npz`` (``rot_partition_sim`` snapshot schema).

    ``npz_data`` is typically the object returned by ``numpy.load(..., allow_pickle=False)``.
    """
    raw = npz_data["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise TypeError(f"metadata_json must decode to a dict, got {type(out).__name__}")
    return out


def wall_definitions_for_dims(
    dx: float,
    dy: float,
    dz: float,
    wall_thickness: float,
    wall_scale: float,
    walls_removed: set[int] | None = None,
):
    """
    Box container walls in world space (same layout as batch unload / rot_partition builders).

    ``walls_removed``: indices 0..3 to omit (0=left -x, 1=right +x, 2=front -y, 3=back +y).

    Returns list of ``(position_vec3, hx, hy, hz)`` for ``add_shape_box`` static walls.
    """
    rm = walls_removed if walls_removed else set()
    wall_h = dz * 1.1
    whz = wall_h / 2.0
    cx, cy = dx / 2.0, dy / 2.0
    chx = dx * wall_scale / 2.0
    chy = dy * wall_scale / 2.0
    wt = wall_thickness
    wall_defs_all = [
        (wp.vec3(cx - chx - wt, cy, whz), wt, chy + wt, whz),
        (wp.vec3(cx + chx + wt, cy, whz), wt, chy + wt, whz),
        (wp.vec3(cx, cy - chy - wt, whz), chx + wt, wt, whz),
        (wp.vec3(cx, cy + chy + wt, whz), chx + wt, wt, whz),
    ]
    return [wd for i, wd in enumerate(wall_defs_all) if i not in rm]


def estimate_rigid_contact_max(
    total_bodies: int,
    contacts_per_body: int,
    user_override: int,
    min_rigid_contact_max: int,
) -> int:
    """
    Global rigid contact buffer size for Newton `CollisionPipeline` (`rigid_contact_max`).

    If ``user_override > 0``, returns it. Otherwise ``max(total_bodies * contacts_per_body, min_rigid_contact_max)``.
    """
    if user_override > 0:
        return user_override
    if contacts_per_body <= 0:
        raise ValueError(f"contacts_per_body must be positive, got {contacts_per_body}.")
    return max(total_bodies * contacts_per_body, min_rigid_contact_max)


def compute_world_speed_stats_numpy(
    body_world_start_np: np.ndarray,
    body_qd_np: np.ndarray,
    body_diag_np: np.ndarray,
    world_speed_threshold_np: np.ndarray,
    world_id: int,
) -> tuple[float, float, float, float]:
    """
    For one world: linear threshold (m/s), max |v|, max |omega|, max equivalent speed.

    ``max_equiv`` uses sqrt(|v|^2 + |omega|^2 * (0.5 * diag)^2) per body, then max over bodies
    (diagnostic; settle uses ``check_body_stability_lin_ang`` with separate lin / ang thresholds).
    """
    w_start = int(body_world_start_np[world_id])
    w_end = int(body_world_start_np[world_id + 1])
    speed_thr = float(world_speed_threshold_np[world_id])
    if w_start >= w_end:
        return speed_thr, 0.0, 0.0, 0.0
    sl = body_qd_np[w_start:w_end]
    di = body_diag_np[w_start:w_end]
    lv = sl[:, :3]
    av = sl[:, 3:6]
    lv_sq = np.sum(lv * lv, axis=1)
    av_sq = np.sum(av * av, axis=1)
    r = 0.5 * di
    equiv_sq = lv_sq + av_sq * r * r
    max_lin = float(np.sqrt(np.max(lv_sq)))
    max_ang = float(np.sqrt(np.max(av_sq)))
    max_equiv = float(np.sqrt(np.max(equiv_sq)))
    return speed_thr, max_lin, max_ang, max_equiv
