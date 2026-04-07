"""Discrete rotation + anisotropic in-plane scaling for partitioned boxes (one thread per box slot)."""

import warp as wp


@wp.func
def unit_cube_sign(vi: int, axis: int) -> float:
    s = float(-1.0)
    if axis == 0:
        if vi == 1 or vi == 2 or vi == 5 or vi == 6:
            s = 1.0
    elif axis == 1:
        if vi == 2 or vi == 3 or vi == 6 or vi == 7:
            s = 1.0
    else:
        if vi >= 4:
            s = 1.0
    return s


@wp.func
def build_rotation(axis: wp.vec3, angle: float) -> wp.mat33:
    c = wp.cos(angle)
    s = wp.sin(angle)
    t = 1.0 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    return wp.mat33(
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    )


@wp.kernel
def discrete_rotation_kernel(
    boxes: wp.array3d(dtype=wp.float32),
    box_counts: wp.array(dtype=wp.int32),
    user_shrink: float,
    max_boxes: int,
    seed: int,
    isotropic: int,
    out_verts: wp.array3d(dtype=wp.float32),
    out_centers: wp.array(dtype=wp.vec3f),
    out_quats: wp.array(dtype=wp.quatf),
    out_half_extents: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    env = tid // max_boxes
    box_idx = tid % max_boxes

    if box_idx >= box_counts[env]:
        return

    x0 = boxes[env, box_idx, 0]
    x1 = boxes[env, box_idx, 1]
    y0 = boxes[env, box_idx, 2]
    y1 = boxes[env, box_idx, 3]
    z0 = boxes[env, box_idx, 4]
    z1 = boxes[env, box_idx, 5]

    center = wp.vec3((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)
    h = wp.vec3((x1 - x0) * 0.5, (y1 - y0) * 0.5, (z1 - z0) * 0.5)

    state = wp.rand_init(seed, tid)

    rot_axis = wp.vec3(0.0, 0.0, 1.0)
    angle = float(0.0)

    if isotropic != 0:
        choice = wp.randi(state) % 10
        if choice == 0:
            rot_axis = wp.vec3(0.0, 0.0, 1.0)
            angle = 0.0
        else:
            c = choice - 1
            axis_choice = c // 3
            ai = c % 3
            rot_axis = wp.vec3(1.0, 0.0, 0.0)
            if axis_choice == 1:
                rot_axis = wp.vec3(0.0, 1.0, 0.0)
            elif axis_choice == 2:
                rot_axis = wp.vec3(0.0, 0.0, 1.0)
            if ai == 0:
                angle = wp.pi / 4.0
            elif ai == 1:
                angle = wp.pi / 2.0
            else:
                angle = 3.0 * wp.pi / 4.0
    else:
        a0 = h[0] * h[1]
        a1 = h[0] * h[2]
        a2 = h[1] * h[2]

        p_norm = wp.vec3(1.0, 0.0, 0.0)
        if a1 >= a0 and a1 >= a2:
            p_norm = wp.vec3(0.0, 1.0, 0.0)
        elif a2 >= a0 and a2 >= a1:
            p_norm = wp.vec3(0.0, 0.0, 1.0)

        world_z = wp.vec3(0.0, 0.0, 1.0)
        rot_axis = world_z
        if wp.abs(wp.dot(p_norm, world_z)) <= 0.99:
            rot_axis = wp.normalize(wp.cross(p_norm, world_z))

        ai = wp.randi(state) % 4
        if ai == 1:
            angle = wp.pi / 4.0
        elif ai == 2:
            angle = wp.pi / 2.0
        elif ai == 3:
            angle = 3.0 * wp.pi / 4.0

    mat = build_rotation(rot_axis, angle)

    s_lo = float(0.0)
    s_hi = float(1.0)

    for _bs in range(15):
        s_mid = (s_lo + s_hi) * 0.5
        fits = int(1)

        for vi in range(8):
            v_local = wp.vec3(
                unit_cube_sign(vi, 0) * h[0],
                unit_cube_sign(vi, 1) * h[1],
                unit_cube_sign(vi, 2) * h[2],
            )
            d = wp.dot(v_local, rot_axis)
            proj = d * rot_axis
            perp = v_local - proj
            v_scaled = proj + s_mid * perp
            v_rot = mat * v_scaled

            if wp.abs(v_rot[0]) > h[0] + 1.0e-6:
                fits = 0
            if wp.abs(v_rot[1]) > h[1] + 1.0e-6:
                fits = 0
            if wp.abs(v_rot[2]) > h[2] + 1.0e-6:
                fits = 0

        if fits == 1:
            s_lo = s_mid
        else:
            s_hi = s_mid

    s_plane = s_lo * user_shrink

    max_hx = float(0.0)
    max_hy = float(0.0)
    max_hz = float(0.0)

    for vi in range(8):
        v_local = wp.vec3(
            unit_cube_sign(vi, 0) * h[0],
            unit_cube_sign(vi, 1) * h[1],
            unit_cube_sign(vi, 2) * h[2],
        )
        d = wp.dot(v_local, rot_axis)
        proj = d * rot_axis
        perp = v_local - proj
        v_scaled = proj + s_plane * perp

        ax = wp.abs(v_scaled[0])
        ay = wp.abs(v_scaled[1])
        az = wp.abs(v_scaled[2])
        if ax > max_hx:
            max_hx = ax
        if ay > max_hy:
            max_hy = ay
        if az > max_hz:
            max_hz = az

        v_rot = mat * v_scaled + center

        out_verts[tid, vi, 0] = v_rot[0]
        out_verts[tid, vi, 1] = v_rot[1]
        out_verts[tid, vi, 2] = v_rot[2]

    half_a = angle * 0.5
    sa = wp.sin(half_a)
    out_centers[tid] = center
    out_quats[tid] = wp.quatf(rot_axis[0] * sa, rot_axis[1] * sa, rot_axis[2] * sa, wp.cos(half_a))
    out_half_extents[tid] = wp.vec3f(max_hx, max_hy, max_hz)
