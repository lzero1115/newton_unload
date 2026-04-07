import argparse
import math
import time
import numpy as np
import warp as wp
from kernels.trivial_partition_kernel import partition_batched_kernel
from kernels.rot_partition_kernel import discrete_rotation_kernel

wp.init()

def rot_batched_partition(num_envs, dims, n_target, min_ratio, shrink, seed, device, isotropic=False):
    dims_np = np.asarray(dims, dtype=np.float32)
    if dims_np.ndim == 1:
        dims_np = np.tile(dims_np[None, :], (num_envs, 1))

    max_boxes = n_target

    boxes_np = np.zeros((num_envs, max_boxes, 6), dtype=np.float32)
    boxes_np[:, 0, 1] = dims_np[:, 0]
    boxes_np[:, 0, 3] = dims_np[:, 1]
    boxes_np[:, 0, 5] = dims_np[:, 2]

    counts_np = np.ones(num_envs, dtype=np.int32)

    t0 = time.perf_counter()

    wp_boxes = wp.array(boxes_np, dtype=wp.float32, device=device)
    wp_counts = wp.array(counts_np, dtype=wp.int32, device=device)
    wp_dims = wp.array(dims_np, dtype=wp.float32, device=device)

    t1 = time.perf_counter()

    wp.launch(
        kernel=partition_batched_kernel,
        dim=num_envs,
        inputs=[wp_boxes, wp_counts, wp_dims, n_target, min_ratio, seed],
        device=device,
    )
    wp.synchronize()

    t2 = time.perf_counter()

    total_threads = num_envs * max_boxes
    wp_out_verts = wp.zeros((total_threads, 8, 3), dtype=wp.float32, device=device)
    wp_out_centers = wp.zeros(total_threads, dtype=wp.vec3f, device=device)
    wp_out_quats = wp.zeros(total_threads, dtype=wp.quatf, device=device)
    wp_out_half_extents = wp.zeros(total_threads, dtype=wp.vec3f, device=device)

    rot_seed = seed + 1000000
    wp.launch(
        kernel=discrete_rotation_kernel,
        dim=total_threads,
        inputs=[
            wp_boxes,
            wp_counts,
            shrink,
            max_boxes,
            rot_seed,
            int(isotropic),
            wp_out_verts,
            wp_out_centers,
            wp_out_quats,
            wp_out_half_extents,
        ],
        device=device,
    )
    wp.synchronize()

    t3 = time.perf_counter()

    result = (
        wp_boxes.numpy(),
        wp_counts.numpy(),
        wp_out_verts.numpy(),
        dims_np,
        max_boxes,
        wp_out_centers.numpy(),
        wp_out_quats.numpy(),
        wp_out_half_extents.numpy(),
    )

    t4 = time.perf_counter()

    print(f"[run] upload:     {(t1-t0)*1000:7.2f} ms")
    print(f"[run] kernel 1:   {(t2-t1)*1000:7.2f} ms  (batched partition)")
    mode = "discrete rotation (isotropic, 10 poses)" if isotropic else "discrete rotation (z prior)"
    print(f"[run] kernel 2:   {(t3-t2)*1000:7.2f} ms  ({mode})")
    print(f"[run] download:   {(t4-t3)*1000:7.2f} ms")
    print(f"[run] total:      {(t4-t0)*1000:7.2f} ms")

    return result

def visualize(ps, out_boxes, out_counts, out_verts, dims_np, max_boxes, show_containers, seed):
    ps.init()
    ps.set_up_dir("z_up")

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [6, 5, 4], [7, 6, 4],
        [1, 5, 6], [1, 6, 2], [0, 4, 7], [0, 7, 3],
        [3, 2, 6], [3, 6, 7], [1, 0, 4], [1, 4, 5],
    ])

    num_envs = len(out_counts)
    cols = int(math.ceil(math.sqrt(num_envs)))
    max_dims = np.max(dims_np, axis=0)
    spacing = max_dims + np.array([0.25, 0.25, 0.25])

    rng = np.random.default_rng(seed + 1)

    for env in range(num_envs):
        row = env // cols
        col = env % cols
        offset = np.array([col * spacing[0], row * spacing[1], 0.0], dtype=np.float64)
        count = int(out_counts[env])

        if show_containers:
            env_dims = dims_np[env].astype(np.float64)
            c_center = offset + 0.5 * env_dims
            half = env_dims / 2.0
            cv = np.array([
                c_center + [-half[0], -half[1], -half[2]],
                c_center + [half[0], -half[1], -half[2]],
                c_center + [half[0], half[1], -half[2]],
                c_center + [-half[0], half[1], -half[2]],
                c_center + [-half[0], -half[1], half[2]],
                c_center + [half[0], -half[1], half[2]],
                c_center + [half[0], half[1], half[2]],
                c_center + [-half[0], half[1], half[2]],
            ])
            edges = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ])
            cn = ps.register_curve_network(f"container_{env}", cv, edges)
            cn.set_color((0.6, 0.6, 0.6))
            cn.set_transparency(0.5)
            cn.set_radius(0.002)

        for i in range(count):
            flat_idx = env * max_boxes + i
            verts = out_verts[flat_idx].astype(np.float64) + offset
            mesh = ps.register_surface_mesh(f"env{env}_box{i}", verts, faces)
            mesh.set_color(tuple(rng.random(3)))
            #mesh.set_edge_width(1.0)

    print(f"Rendered {num_envs} environments, {sum(out_counts)} total rotated boxes.")
    ps.show()



def main():
    parser = argparse.ArgumentParser(description="Batched partition + discrete rotation (Warp)")
    parser.add_argument("--num_envs","--ne", type=int, default=16)
    parser.add_argument("--dims", type=float, nargs=3, default=[0.6, 0.4, 0.3])
    parser.add_argument("--n","--nb", type=int, default=50)
    parser.add_argument("--min_ratio", type=float, default=0.04)
    parser.add_argument("--shrink", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show_containers", action="store_true")
    parser.add_argument("--vis", action="store_true", help="Launch Polyscope visualization.")
    parser.add_argument(
        "--isotropic",
        action="store_true",
        dest="isotropic",
        help=(
            "Random pose init: 10 distinct poses — one identity (0°) plus 3 world axes × "
            "{π/4,π/2,3π/4} (0° about X/Y/Z is the same, counted once). "
            "Default mode uses largest-face normal × world-z prior and angles in {0,π/4,π/2,3π/4}. "
            "--ignore-z is a deprecated alias."
        ),
    )
    args = parser.parse_args()

    t_run_start = time.perf_counter()
    (out_boxes, out_counts, out_verts, dims_np, max_boxes,
     out_centers, out_quats, out_half_extents) = rot_batched_partition(
        num_envs=args.num_envs,
        dims=args.dims,
        n_target=args.n,
        min_ratio=args.min_ratio,
        shrink=args.shrink,
        seed=args.seed,
        device=args.device,
        isotropic=args.isotropic,
    )
    t_run_end = time.perf_counter()
    print(f"\n[main] run total:       {(t_run_end-t_run_start)*1000:7.2f} ms")

    if args.vis:
        import polyscope as ps

        t_vis_start = time.perf_counter()
        visualize(ps, out_boxes, out_counts, out_verts, dims_np, max_boxes,
                  args.show_containers, args.seed)
        t_vis_end = time.perf_counter()
        print(f"[main] visualize total: {(t_vis_end-t_vis_start)*1000:7.2f} ms")


if __name__ == "__main__":
    main()
