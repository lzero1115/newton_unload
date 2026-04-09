"""GPU extraction of axis-aligned box corners in world space (visualization).

Both kernels mirror ``sim_kernels``: 2D ``wp.tid()``, prefix array, ``global_id = w_start + local``,
padding early-out, output slot ``(row * max_per_row + local) * 8``.

- Dynamic: ``body_world_start``, row = ``world_id``; half extents in ``half_extents_batched[slot]`` with
  ``slot = world_id * max_bodies_per_world + body_local_id`` (same as ``rot_batched_partition`` / scene order).
- Static walls: ``static_segment_start`` (currently ``[0, n]``), row = ``seg_id``; extend to one segment per Newton world later.
"""

import warp as wp


@wp.func
def _write_box8(xf: wp.transform, hx: float, hy: float, hz: float, out_base: int, out_verts: wp.array(dtype=wp.vec3)):
    o = out_base
    out_verts[o + 0] = wp.transform_point(xf, wp.vec3(-hx, -hy, -hz))
    out_verts[o + 1] = wp.transform_point(xf, wp.vec3(hx, -hy, -hz))
    out_verts[o + 2] = wp.transform_point(xf, wp.vec3(hx, hy, -hz))
    out_verts[o + 3] = wp.transform_point(xf, wp.vec3(-hx, hy, -hz))
    out_verts[o + 4] = wp.transform_point(xf, wp.vec3(-hx, -hy, hz))
    out_verts[o + 5] = wp.transform_point(xf, wp.vec3(hx, -hy, hz))
    out_verts[o + 6] = wp.transform_point(xf, wp.vec3(hx, hy, hz))
    out_verts[o + 7] = wp.transform_point(xf, wp.vec3(-hx, hy, hz))


@wp.kernel
def extract_box_vertices_world(
    body_q: wp.array(dtype=wp.transform),
    half_extents_batched: wp.array(dtype=wp.vec3),
    body_world_start: wp.array(dtype=wp.int32),
    max_bodies_per_world: int,
    out_verts: wp.array(dtype=wp.vec3),
):
    """Pose from ``body_q[global_id]``; half extents from batched slot (partition / scene order)."""
    world_id, body_local_id = wp.tid()
    w_start = body_world_start[world_id]
    w_count = body_world_start[world_id + 1] - w_start
    if body_local_id >= w_count:
        return
    global_id = w_start + body_local_id
    xf = body_q[global_id]
    slot = world_id * max_bodies_per_world + body_local_id
    he = half_extents_batched[slot]
    out_base = slot * 8
    _write_box8(xf, he[0], he[1], he[2], out_base, out_verts)


@wp.kernel
def extract_static_box_vertices_world(
    xforms: wp.array(dtype=wp.transform),
    half_extents: wp.array(dtype=wp.vec3),
    static_segment_start: wp.array(dtype=wp.int32),
    max_walls_per_segment: int,
    out_verts: wp.array(dtype=wp.vec3),
):
    """Static boxes (walls): ``static_segment_start`` length ``num_segments + 1`` (like ``body_world_start``)."""
    seg_id, wall_local_id = wp.tid()
    w_start = static_segment_start[seg_id]
    w_count = static_segment_start[seg_id + 1] - w_start
    if wall_local_id >= w_count:
        return
    global_id = w_start + wall_local_id
    xf = xforms[global_id]
    he = half_extents[global_id]
    out_base = (seg_id * max_walls_per_segment + wall_local_id) * 8
    _write_box8(xf, he[0], he[1], he[2], out_base, out_verts)
