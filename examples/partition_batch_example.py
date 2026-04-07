import argparse
import math

import numpy as np
import polyscope as ps
import warp as wp
from kernels.trivial_partition_kernel import partition_batched_kernel

wp.init()

def batched_partition(
    num_envs=16,
    dims=(0.6, 0.4, 0.3),
    n_target=50,
    min_ratio=0.05,
    shrink_factor=0.98,
    seed=42,
    device="cuda",
):
    dims_np = np.asarray(dims, dtype=np.float32)
    if dims_np.shape == (3,):
        dims_np = np.tile(dims_np[None, :], (num_envs, 1))
    else:
        assert dims_np.shape == (num_envs, 3)

    max_boxes = n_target

    boxes_np = np.zeros((num_envs, max_boxes, 6), dtype=np.float32)
    boxes_np[:, 0, 1] = dims_np[:, 0]
    boxes_np[:, 0, 3] = dims_np[:, 1]
    boxes_np[:, 0, 5] = dims_np[:, 2]

    counts_np = np.ones((num_envs,), dtype=np.int32)

    wp_boxes = wp.array(boxes_np, dtype=wp.float32, device=device)
    wp_counts = wp.array(counts_np, dtype=wp.int32, device=device)
    wp_dims = wp.array(dims_np, dtype=wp.float32, device=device)

    wp.launch(
        kernel=partition_batched_kernel,
        dim=num_envs,
        inputs=[wp_boxes, wp_counts, wp_dims, n_target, min_ratio, seed],
        device=device,
    )

    wp.synchronize()

    out_boxes = wp_boxes.numpy()
    out_counts = wp_counts.numpy()

    results = []
    for env in range(num_envs):
        env_result = []
        count = int(out_counts[env])

        for i in range(count):
            b = out_boxes[env, i]
            x0, x1, y0, y1, z0, z1 = b.tolist()

            center = np.array([
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (z0 + z1),
            ], dtype=np.float64)

            lengths = np.array([
                max((x1 - x0) * shrink_factor, 1e-6),
                max((y1 - y0) * shrink_factor, 1e-6),
                max((z1 - z0) * shrink_factor, 1e-6),
            ], dtype=np.float64)

            env_result.append((center, lengths))

        results.append(env_result)

    return results, dims_np


def visualize_polyscope(results, dims_np, seed=123):
    ps.init()
    ps.set_up_dir("z_up")

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [6, 5, 4], [7, 6, 4],
        [1, 5, 6], [1, 6, 2], [0, 4, 7], [0, 7, 3],
        [3, 2, 6], [3, 6, 7], [1, 0, 4], [1, 4, 5],
    ])

    num_envs = len(results)
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))

    max_dims = np.max(dims_np, axis=0)
    spacing = max_dims + np.array([0.25, 0.25, 0.25], dtype=np.float64)

    rng = np.random.default_rng(seed)

    for env in range(num_envs):
        row = env // cols
        col = env % cols

        offset = np.array([
            col * spacing[0],
            row * spacing[1],
            0.0,
        ], dtype=np.float64)

        env_dims = dims_np[env].astype(np.float64)
        container_center = offset + 0.5 * env_dims

        # Draw container as a wireframe curve network
        half = env_dims / 2.0
        c = container_center
        cv = np.array([
            c + [-half[0], -half[1], -half[2]],
            c + [ half[0], -half[1], -half[2]],
            c + [ half[0],  half[1], -half[2]],
            c + [-half[0],  half[1], -half[2]],
            c + [-half[0], -half[1],  half[2]],
            c + [ half[0], -half[1],  half[2]],
            c + [ half[0],  half[1],  half[2]],
            c + [-half[0],  half[1],  half[2]],
        ])
        edges = np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ])
        cn = ps.register_curve_network(f"container_{env}", cv, edges)
        cn.set_color((0.6, 0.6, 0.6))
        cn.set_radius(0.002)

        for i, (center, lengths) in enumerate(results[env]):
            world_center = center + offset
            half_b = lengths / 2.0
            c_b = world_center
            verts = np.array([
                c_b + [-half_b[0], -half_b[1], -half_b[2]],
                c_b + [ half_b[0], -half_b[1], -half_b[2]],
                c_b + [ half_b[0],  half_b[1], -half_b[2]],
                c_b + [-half_b[0],  half_b[1], -half_b[2]],
                c_b + [-half_b[0], -half_b[1],  half_b[2]],
                c_b + [ half_b[0], -half_b[1],  half_b[2]],
                c_b + [ half_b[0],  half_b[1],  half_b[2]],
                c_b + [-half_b[0],  half_b[1],  half_b[2]],
            ])
            mesh = ps.register_surface_mesh(f"env_{env}_block_{i}", verts, faces)
            mesh.set_color(tuple(rng.random(3)))

    print(f"Rendered {num_envs} environments in a {rows}x{cols} grid.")
    ps.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs","--ne", type=int, default=16)
    parser.add_argument("--dims", type=float, nargs=3, default=[0.6, 0.4, 0.3])
    parser.add_argument("--n","--nb", type=int, default=50)
    parser.add_argument("--min_ratio", type=float, default=0.05)
    parser.add_argument("--shrink", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results, dims_np = batched_partition(
        num_envs=args.num_envs,
        dims=args.dims,
        n_target=args.n,
        min_ratio=args.min_ratio,
        shrink_factor=args.shrink,
        seed=args.seed,
        device=args.device,
    )

    visualize_polyscope(results, dims_np, seed=args.seed + 1)


if __name__ == "__main__":
    main()