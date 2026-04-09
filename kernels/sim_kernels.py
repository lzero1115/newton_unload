"""
Shared Warp kernels for multi-world rigid sims: settle detection and frozen-world poses.

``check_body_stability`` (legacy equiv-speed) and ``check_body_stability_lin_ang`` (separate |v| / |ω|
thresholds; **per-world** unsettled counts) / ``check_body_stability_lin_ang_per_body_unsettled`` (same
criterion, **per-body** 0/1 for e.g. viewer coloring) / ``enforce_frozen_worlds`` take ``active_mask``
(1 = participate).
Use all-ones when every dynamic body counts (e.g. ``rot_partition_sim``); use real masks for
unload flows with removed bodies.

``enforce_removed_bodies`` holds ``active_mask == 0`` bodies at ``frozen_body_q`` (unload / removal).

``inter_steady_metric_weighted_sum_per_world_masked`` / ``zero_world_active_velocities_masked`` take a
per-world ``world_mask`` (length = world count).
"""

import warp as wp


@wp.kernel
def check_body_stability(
    body_world_start: wp.array(dtype=wp.int32),
    world_frozen: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32), # body level mask
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_diag: wp.array(dtype=wp.float32),
    world_speed_threshold: wp.array(dtype=wp.float32),
    world_unsettled: wp.array(dtype=wp.int32),
):
    # Per body: unsettled if sqrt(|v|^2 + |omega|^2 * r^2) >= threshold, r = 0.5*diag (sqrt-free: compare squares).
    # Legacy equiv-speed criterion; prefer ``check_body_stability_lin_ang`` with fixed lin/ang thresholds.
    world_id, body_local_id = wp.tid()
    if world_frozen[world_id] != 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    twist = body_qd[global_id]
    lv = wp.vec3(twist[0], twist[1], twist[2])
    av = wp.vec3(twist[3], twist[4], twist[5])
    diag = body_diag[global_id]
    r = 0.5 * diag
    r2 = r * r
    lv_sq = wp.dot(lv, lv)
    av_sq = wp.dot(av, av)
    equiv_sq = lv_sq + av_sq * r2
    max_speed = world_speed_threshold[world_id]
    thr_sq = max_speed * max_speed

    if equiv_sq >= thr_sq:
        wp.atomic_add(world_unsettled, world_id, 1)


@wp.kernel
def check_body_stability_lin_ang(
    body_world_start: wp.array(dtype=wp.int32),
    world_frozen: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    world_lin_threshold: wp.array(dtype=wp.float32),
    world_ang_threshold: wp.array(dtype=wp.float32),
    world_unsettled: wp.array(dtype=wp.int32),
):
    """Unsettled if |v| >= lin_thr or |ω| >= ang_thr (per-body L2 norms; sqrt-free on squared norms)."""
    world_id, body_local_id = wp.tid()
    if world_frozen[world_id] != 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    twist = body_qd[global_id]
    lv = wp.vec3(twist[0], twist[1], twist[2])
    av = wp.vec3(twist[3], twist[4], twist[5])
    lv_sq = wp.dot(lv, lv)
    av_sq = wp.dot(av, av)
    lin_thr = world_lin_threshold[world_id]
    ang_thr = world_ang_threshold[world_id]
    lin_thr_sq = lin_thr * lin_thr
    ang_thr_sq = ang_thr * ang_thr
    if lv_sq >= lin_thr_sq or av_sq >= ang_thr_sq:
        wp.atomic_add(world_unsettled, world_id, 1)


@wp.kernel
def check_body_stability_lin_ang_per_body_unsettled(
    body_world_start: wp.array(dtype=wp.int32),
    world_frozen: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    world_lin_threshold: wp.array(dtype=wp.float32),
    world_ang_threshold: wp.array(dtype=wp.float32),
    body_unsettled: wp.array(dtype=wp.int32),
):
    """Same pass/fail as ``check_body_stability_lin_ang``, but set ``body_unsettled[global_id]=1`` (else 0).

    Caller must zero ``body_unsettled`` before launch. Threads that return early leave prior zeros
    (e.g. padding locals, inactive bodies). Frozen worlds: no writes (remain zero → \"settled\" for viz).
    """
    world_id, body_local_id = wp.tid()
    if world_frozen[world_id] != 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    twist = body_qd[global_id]
    lv = wp.vec3(twist[0], twist[1], twist[2])
    av = wp.vec3(twist[3], twist[4], twist[5])
    lv_sq = wp.dot(lv, lv)
    av_sq = wp.dot(av, av)
    lin_thr = world_lin_threshold[world_id]
    ang_thr = world_ang_threshold[world_id]
    lin_thr_sq = lin_thr * lin_thr
    ang_thr_sq = ang_thr * ang_thr
    if lv_sq >= lin_thr_sq or av_sq >= ang_thr_sq:
        body_unsettled[global_id] = 1
    else:
        body_unsettled[global_id] = 0


@wp.kernel
def capture_frozen_body_state(
    body_world_start: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    frozen_body_q: wp.array(dtype=wp.transform),
):
    world_id, body_local_id = wp.tid()
    if world_mask[world_id] == 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    pose = body_q[global_id]
    frozen_body_q[global_id] = pose


@wp.kernel
def enforce_frozen_worlds(
    body_world_start: wp.array(dtype=wp.int32),
    world_frozen: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32),
    frozen_body_q: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
): # frozen whole worlds
    world_id, body_local_id = wp.tid()
    if world_frozen[world_id] == 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    body_q[global_id] = frozen_body_q[global_id]
    body_qd[global_id] = wp.spatial_vector()


@wp.kernel
def enforce_removed_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    active_mask: wp.array(dtype=wp.int32),
    frozen_body_q: wp.array(dtype=wp.transform),
    body_world_start: wp.array(dtype=wp.int32),
): # frozen removed bodies
    """Bodies with ``active_mask == 0`` (removed) are held at ``frozen_body_q`` with zero twist."""
    world_id, body_local_id = wp.tid()
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] != 0:
        return
    body_q[global_id] = frozen_body_q[global_id]
    body_qd[global_id] = wp.spatial_vector()

# TODO: metric modify
@wp.kernel
def inter_steady_metric_weighted_sum_per_world_masked(
    body_world_start: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q_curr: wp.array(dtype=wp.transform),
    half_extents: wp.array(dtype=wp.vec3),
    body_diag: wp.array(dtype=wp.float32),
    rot_scale: float,
    world_mask: wp.array(dtype=wp.int32),
    out_per_world: wp.array(dtype=wp.float32),
):
    """Accumulate volume-weighted inter-steady metric into ``out_per_world[world_id]`` if ``world_mask[world_id]``."""
    world_id, body_local_id = wp.tid()
    if world_mask[world_id] == 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    t0 = wp.transform_get_translation(body_q_prev[global_id])
    t1 = wp.transform_get_translation(body_q_curr[global_id])
    dp = t1 - t0
    trans = wp.length(dp)
    q0 = wp.transform_get_rotation(body_q_prev[global_id])
    q1 = wp.transform_get_rotation(body_q_curr[global_id])
    d = wp.abs(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3])
    d = wp.min(d, 1.0)
    ang = 2.0 * wp.acos(d)
    he = half_extents[global_id]
    vol = 8.0 * he[0] * he[1] * he[2]
    r_char = 0.5 * body_diag[global_id]
    contrib = vol * (trans + rot_scale * r_char * ang)
    wp.atomic_add(out_per_world, world_id, contrib)


@wp.kernel
def zero_world_active_velocities_masked(
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    active_mask: wp.array(dtype=wp.int32),
    body_world_start: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.int32),
):
    """Zero twist for active bodies in worlds with ``world_mask[world_id] != 0``."""
    world_id, body_local_id = wp.tid()
    if world_mask[world_id] == 0:
        return
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    if active_mask[global_id] == 0:
        return
    body_qd[global_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
