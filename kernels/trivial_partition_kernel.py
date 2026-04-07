import warp as wp

# easy box volume partition kernel, inspired by ranked reward

@wp.kernel
def partition_batched_kernel(
    boxes: wp.array3d(dtype=wp.float32),  # [num_envs, max_boxes, 6]
    box_counts: wp.array(dtype=wp.int32),  # [num_envs]
    dims: wp.array2d(dtype=wp.float32),  # [num_envs, 3]
    n_target: int,
    min_ratio: float,
    seed: int,
):
    env = wp.tid()

    dim_x = dims[env, 0]
    dim_y = dims[env, 1]
    dim_z = dims[env, 2]

    diag = wp.sqrt(dim_x * dim_x + dim_y * dim_y + dim_z * dim_z)
    min_dim = diag * min_ratio

    count = int(box_counts[env])

    while count < n_target:
        best_i = int(-1)
        best_axis = int(-1)
        best_vol = float(-1.0)

        for i in range(count):
            x0 = boxes[env, i, 0]
            x1 = boxes[env, i, 1]
            y0 = boxes[env, i, 2]
            y1 = boxes[env, i, 3]
            z0 = boxes[env, i, 4]
            z1 = boxes[env, i, 5]

            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0

            axis = int(-1)
            longest = float(-1.0)

            if dx >= 2.0 * min_dim and dx > longest:
                axis = int(0)
                longest = dx
            if dy >= 2.0 * min_dim and dy > longest:
                axis = int(1)
                longest = dy
            if dz >= 2.0 * min_dim and dz > longest:
                axis = int(2)
                longest = dz

            if axis >= 0:
                vol = dx * dy * dz
                if vol > best_vol:
                    best_vol = vol
                    best_i = i
                    best_axis = axis

        if best_i < 0:
            break

        x0 = boxes[env, best_i, 0]
        x1 = boxes[env, best_i, 1]
        y0 = boxes[env, best_i, 2]
        y1 = boxes[env, best_i, 3]
        z0 = boxes[env, best_i, 4]
        z1 = boxes[env, best_i, 5]

        state = wp.rand_init(seed, env * n_target + count)
        u = wp.randf(state)

        new_idx = count

        for j in range(6):
            boxes[env, new_idx, j] = boxes[env, best_i, j]

        if best_axis == 0:
            low = x0 + min_dim
            high = x1 - min_dim
            split = low if high <= low else low + (high - low) * u

            boxes[env, best_i, 1] = split
            boxes[env, new_idx, 0] = split

        elif best_axis == 1:
            low = y0 + min_dim
            high = y1 - min_dim
            split = low if high <= low else low + (high - low) * u

            boxes[env, best_i, 3] = split
            boxes[env, new_idx, 2] = split

        else:
            low = z0 + min_dim
            high = z1 - min_dim
            split = low if high <= low else low + (high - low) * u

            boxes[env, best_i, 5] = split
            boxes[env, new_idx, 4] = split

        count += 1

    box_counts[env] = count