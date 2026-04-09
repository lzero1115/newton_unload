# Simplified rot_partition_sim: exactly one Newton world, no Newton viewer.
# Interactive viz: ``utils.ps_viewer.run_polyscope_rot_partition`` (Polyscope).
# Use --headless for batch / snapshot only.
#
# Run from unload_clean root:
#   python examples/ps_partition_sim.py
#   python examples/ps_partition_sim.py --headless --save-snapshot out.npz
#   python examples/ps_partition_sim.py --partition-seed 123

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from configs import configs_sim as cs
from examples.partition_rot_batch_example import rot_batched_partition
from kernels.sim_kernels import (
    capture_frozen_body_state,
    check_body_stability_lin_ang,
    check_body_stability_lin_ang_per_body_unsettled,
    enforce_frozen_worlds,
)
from utils.create_solver import create_solver
from utils.world_info import estimate_rigid_contact_max, wall_definitions_for_dims

_NUM_WORLDS = 1


def _validate_partition_for_solver(out_counts: np.ndarray, n_target: int, solver_type: str) -> None:
    if solver_type == "mujoco" and int(out_counts[0]) != int(n_target):
        raise ValueError(
            f"MuJoCo requires homogeneous box count; got {int(out_counts[0])} vs target {n_target}."
        )


class PsPartitionSim:
    """Single-world rotated partition scene; physics matches rot_partition_sim (no Newton viewer)."""

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.solver_type = str(args.solver)
        self.fps = int(cs.FPS)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = max(1, int(cs.SIM_SUBSTEPS))
        self.sim_dt = self.frame_dt / self.sim_substeps

        n_target = int(args.nb)
        if n_target < 1:
            raise ValueError(f"--nb must be >= 1, got {n_target}")
        self.n_target = n_target
        self.partition_seed = int(getattr(args, "partition_seed", cs.SEED))

        self.settle_steps_required = int(cs.SETTLE_STEPS)
        self.settle_check_interval = int(cs.SETTLE_CHECK_INTERVAL)
        self.frame_count = 0
        self.snapshot_path = str(args.save_snapshot or "").strip()
        self.snapshot_saved = False

        self._walls_removed: set[int] = set()
        for wi in args.remove_wall or []:
            i = int(wi)
            if i not in (0, 1, 2, 3):
                raise ValueError(
                    f"--remove-wall indices must be 0..3 (0=left -x, 1=right +x, 2=front -y, 3=back +y), got {i}"
                )
            self._walls_removed.add(i)

        dims_list = [float(x) for x in args.dims]
        if any(d <= 0.0 for d in dims_list):
            raise ValueError(f"--dims components must be positive, got {dims_list}")

        print(f"[ps_partition_sim] Single world, partition target {n_target} boxes...")
        (_, out_counts, _, _, partition_max_boxes, out_centers, out_quats, out_half_extents) = (
            rot_batched_partition(
                num_envs=_NUM_WORLDS,
                dims=dims_list,
                n_target=n_target,
                min_ratio=cs.MIN_RATIO,
                shrink=cs.SHRINK,
                seed=self.partition_seed,
                device=args.device,
                isotropic=bool(cs.ISOTROPIC),
            )
        )
        _validate_partition_for_solver(out_counts, n_target, self.solver_type)
        self.partition_max_boxes = int(partition_max_boxes)
        self.partition_num_envs = int(_NUM_WORLDS)
        self.partition_half_extents_batched_np = np.asarray(out_half_extents, dtype=np.float32).reshape(
            -1, 3
        )
        _pe = self.partition_num_envs * self.partition_max_boxes
        if int(self.partition_half_extents_batched_np.shape[0]) != _pe:
            raise ValueError(
                f"partition half rows {self.partition_half_extents_batched_np.shape[0]} != "
                f"partition_num_envs * partition_max_boxes = {_pe}"
            )

        num_envs = _NUM_WORLDS
        main_builder = newton.ModelBuilder()
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(main_builder)
        except Exception as exc:
            print(f"[warn] Could not register MuJoCo custom attributes: {exc}")

        main_builder.rigid_gap = cs.DEFAULT_RIGID_GAP
        main_builder.default_shape_cfg.gap = cs.DEFAULT_RIGID_GAP
        main_builder.default_shape_cfg.ke = cs.MUJOCO_CONTACT_KE
        main_builder.default_shape_cfg.kd = cs.MUJOCO_CONTACT_KD

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=cs.GROUND_MU,
            kd=cs.GROUND_KD,
            mu_torsional=cs.GROUND_MU_TORSIONAL,
            mu_rolling=cs.GROUND_MU_ROLLING,
            gap=cs.DEFAULT_RIGID_GAP,
            ke=cs.MUJOCO_CONTACT_KE,
        )
        main_builder.add_shape_plane(
            body=-1,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=ground_cfg,
        )

        box_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=cs.BOX_MU,
            kd=cs.BOX_KD,
            mu_torsional=cs.BOX_MU_TORSIONAL,
            mu_rolling=cs.BOX_MU_ROLLING,
            gap=cs.DEFAULT_RIGID_GAP,
            ke=cs.MUJOCO_CONTACT_KE,
        )
        wall_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=cs.WALL_MU,
            kd=cs.WALL_KD,
            mu_torsional=cs.WALL_MU_TORSIONAL,
            mu_rolling=cs.WALL_MU_ROLLING,
            gap=cs.DEFAULT_RIGID_GAP,
            ke=cs.MUJOCO_CONTACT_KE,
        )

        dx, dy, dz = float(dims_list[0]), float(dims_list[1]), float(dims_list[2])
        wall_defs = wall_definitions_for_dims(
            dx, dy, dz, float(cs.WALL_THICKNESS), float(cs.WALL_SCALE), walls_removed=self._walls_removed
        )

        body_diagonals: list[float] = []
        body_half_extents: list[tuple[float, float, float]] = []
        world_speed_thresholds: list[float] = []

        env_builder = newton.ModelBuilder()
        env_builder.rigid_gap = cs.DEFAULT_RIGID_GAP
        env_builder.default_shape_cfg.gap = cs.DEFAULT_RIGID_GAP
        env_builder.default_shape_cfg.ke = cs.MUJOCO_CONTACT_KE
        env_builder.default_shape_cfg.kd = cs.MUJOCO_CONTACT_KD

        for pos, whx, why, whz_w in wall_defs:
            env_builder.add_shape_box(
                body=-1,
                xform=wp.transform(pos, wp.quat_identity()),
                hx=whx,
                hy=why,
                hz=whz_w,
                cfg=wall_shape_cfg,
            )

        count = int(out_counts[0])
        for i in range(count):
            flat = i
            c = out_centers[flat]
            q = out_quats[flat]
            he = out_half_extents[flat]
            spawn_pos = wp.vec3(float(c[0]), float(c[1]), float(c[2]) + float(cs.DROP_HEIGHT))
            quat = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            body = env_builder.add_body(xform=wp.transform(p=spawn_pos, q=quat))
            env_builder.add_shape_box(
                body,
                hx=float(he[0]),
                hy=float(he[1]),
                hz=float(he[2]),
                cfg=box_shape_cfg,
            )
            body_half_extents.append((float(he[0]), float(he[1]), float(he[2])))
            diag = 2.0 * math.sqrt(float(he[0]) ** 2 + float(he[1]) ** 2 + float(he[2]) ** 2)
            body_diagonals.append(diag)

        main_builder.add_world(env_builder)
        world_speed_thresholds.append(float(cs.SETTLE_LINEAR_SPEED_MPS))

        self.model = main_builder.finalize(device=args.device)

        if self.solver_type == "mujoco":
            solimp_list = list(cs.MUJOCO_SOLIMP)
            arr = self.model.mujoco.geom_solimp.numpy()
            arr[:] = np.asarray(solimp_list, dtype=arr.dtype)
            print(f"Applied MuJoCo geom_solimp={solimp_list}")

        total_bodies = int(out_counts[0])
        rigid_contact_max = estimate_rigid_contact_max(
            total_bodies=total_bodies,
            contacts_per_body=cs.DEFAULT_CONTACTS_PER_BODY,
            user_override=cs.RIGID_CONTACT_MAX,
            min_rigid_contact_max=cs.MIN_RIGID_CONTACT_MAX,
        )
        self.model.rigid_contact_max = rigid_contact_max
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=rigid_contact_max,
        )
        self.contacts = self.collision_pipeline.contacts()

        nconmax = cs.NCONMAX if cs.NCONMAX > 0 else max(rigid_contact_max, 100)
        njmax = cs.NJMAX if cs.NJMAX > 0 else nconmax * 3
        self.nconmax = nconmax
        self.njmax = njmax

        self.solver = create_solver(
            self.model,
            self.solver_type,
            nconmax=nconmax,
            njmax=njmax,
            xpbd_iterations=cs.XPBD_ITERATIONS,
            xpbd_contact_relaxation=cs.XPBD_CONTACT_RELAXATION,
            xpbd_angular_damping=cs.ANGULAR_DAMPING,
            xpbd_enable_restitution=cs.XPBD_ENABLE_RESTITUTION,
            mujoco_iterations=cs.MUJOCO_ITERATIONS,
            mujoco_ls_iterations=cs.MUJOCO_LS_ITERATIONS,
            mujoco_solver=cs.MUJOCO_SOLVER,
            mujoco_integrator=cs.MUJOCO_INTEGRATOR,
            mujoco_cone=cs.MUJOCO_CONE,
            mujoco_impratio=cs.MUJOCO_IMPRATIO,
            mujoco_tolerance=cs.MUJOCO_TOLERANCE,
            mujoco_ls_tolerance=cs.MUJOCO_LS_TOLERANCE,
            mujoco_update_data_interval=cs.MUJOCO_UPDATE_DATA_INTERVAL,
            mujoco_use_contacts=cs.MUJOCO_USE_CONTACTS,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.body_world_start_np = self.model.body_world_start.numpy()
        body_counts = np.diff(self.body_world_start_np)
        self.max_bodies_per_world = int(np.max(body_counts)) if len(body_counts) > 0 else 1
        self.box_counts_per_world = np.asarray(out_counts, dtype=np.int32)
        self.dims_np = np.asarray(dims_list, dtype=np.float32)
        self.body_half_extents_np = np.asarray(body_half_extents, dtype=np.float32)
        self.body_diag_np = np.asarray(body_diagonals, dtype=np.float32)
        self.body_diag = wp.array(self.body_diag_np, dtype=wp.float32, device=args.device)
        self.active_mask_gpu = wp.array(
            np.ones(self.model.body_count, dtype=np.int32), dtype=wp.int32, device=args.device
        )
        self.world_speed_threshold = wp.array(
            np.asarray(world_speed_thresholds, dtype=np.float32), dtype=wp.float32, device=args.device
        )
        ang_thr = float(cs.SETTLE_ANGULAR_SPEED_RAD)
        self.world_angular_threshold = wp.array(
            np.full(num_envs, ang_thr, dtype=np.float32), dtype=wp.float32, device=args.device
        )
        self.frozen_body_q = wp.clone(self.state_0.body_q)
        self.world_unsettled = wp.zeros(num_envs, dtype=wp.int32)
        self.body_instant_unsettled = wp.zeros(
            int(self.model.body_count), dtype=wp.int32, device=args.device
        )
        self.worlds_settled = np.zeros(num_envs, dtype=bool)
        self.world_frozen_np = np.zeros(num_envs, dtype=np.int32)
        self.world_frozen = wp.zeros(num_envs, dtype=wp.int32, device=args.device)
        self.settle_consecutive = np.zeros(num_envs, dtype=np.int32)
        self.all_settled = False
        self._wall_clock_start: float | None = None

        mujoco_solimp_list = [float(x) for x in cs.MUJOCO_SOLIMP]
        self.snapshot_metadata = {
            "snapshot_version": 1,
            "world_count": int(num_envs),
            "dims": [float(v) for v in self.dims_np.tolist()],
            "solver_type": str(self.solver_type),
            "seed": int(self.partition_seed),
            "partition_seed": int(self.partition_seed),
            "target_boxes_per_world": int(n_target),
            "box_counts_per_world": [int(v) for v in self.box_counts_per_world.tolist()],
            "min_ratio": float(cs.MIN_RATIO),
            "shrink": float(cs.SHRINK),
            "isotropic": bool(cs.ISOTROPIC),
            "fps": int(self.fps),
            "sim_substeps": int(self.sim_substeps),
            "sim_dt": float(self.sim_dt),
            "drop_height": float(cs.DROP_HEIGHT),
            "rigid_gap": float(cs.DEFAULT_RIGID_GAP),
            "mujoco_contact_ke": float(cs.MUJOCO_CONTACT_KE),
            "mujoco_contact_kd": float(cs.MUJOCO_CONTACT_KD),
            "mujoco_solimp": mujoco_solimp_list,
            "ground_mu": float(cs.GROUND_MU),
            "ground_kd": float(cs.GROUND_KD),
            "ground_mu_torsional": float(cs.GROUND_MU_TORSIONAL),
            "ground_mu_rolling": float(cs.GROUND_MU_ROLLING),
            "wall_mu": float(cs.WALL_MU),
            "wall_kd": float(cs.WALL_KD),
            "wall_mu_torsional": float(cs.WALL_MU_TORSIONAL),
            "wall_mu_rolling": float(cs.WALL_MU_ROLLING),
            "box_mu": float(cs.BOX_MU),
            "box_kd": float(cs.BOX_KD),
            "box_mu_torsional": float(cs.BOX_MU_TORSIONAL),
            "box_mu_rolling": float(cs.BOX_MU_ROLLING),
            "wall_thickness": float(cs.WALL_THICKNESS),
            "wall_scale": float(cs.WALL_SCALE),
            "walls_removed": [int(x) for x in sorted(self._walls_removed)],
            "xpbd_iterations": int(cs.XPBD_ITERATIONS),
            "xpbd_contact_relaxation": float(cs.XPBD_CONTACT_RELAXATION),
            "xpbd_angular_damping": float(cs.ANGULAR_DAMPING),
            "angular_damping": float(cs.ANGULAR_DAMPING),
            "mujoco_iterations": int(cs.MUJOCO_ITERATIONS),
            "mujoco_ls_iterations": int(cs.MUJOCO_LS_ITERATIONS),
            "mujoco_solver": str(cs.MUJOCO_SOLVER),
            "mujoco_integrator": str(cs.MUJOCO_INTEGRATOR),
            "mujoco_cone": str(cs.MUJOCO_CONE),
            "mujoco_impratio": float(cs.MUJOCO_IMPRATIO),
            "mujoco_tolerance": float(cs.MUJOCO_TOLERANCE),
            "mujoco_ls_tolerance": float(cs.MUJOCO_LS_TOLERANCE),
            "mujoco_update_data_interval": int(cs.MUJOCO_UPDATE_DATA_INTERVAL),
            "mujoco_use_contacts": bool(cs.MUJOCO_USE_CONTACTS),
            "rigid_contact_max": int(rigid_contact_max),
            "contacts_per_body": int(cs.DEFAULT_CONTACTS_PER_BODY),
            "nconmax": int(self.nconmax),
            "njmax": int(self.njmax),
            "settle_linear_speed_mps": float(cs.SETTLE_LINEAR_SPEED_MPS),
            "settle_angular_speed_rad": float(cs.SETTLE_ANGULAR_SPEED_RAD),
            "settle_steps": int(self.settle_steps_required),
            "settle_check_interval": int(self.settle_check_interval),
            "settle_consecutive_unit": "stability_evaluations",
            "settle_evaluation_every_n_frames": int(self.settle_check_interval),
            "config_module": "configs.configs_sim",
            "ps_partition_sim": True,
        }

        self._box_body_count = count

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate_substeps()
            self.graph = cap.graph
        else:
            self.graph = None

        print(
            f"Newton model ready (solver={self.solver_type}, bodies={total_bodies}, "
            f"rigid_contact_max={rigid_contact_max})."
        )

    @property
    def walls_removed(self) -> frozenset[int]:
        return frozenset(self._walls_removed)

    @property
    def box_body_count(self) -> int:
        return self._box_body_count

    @property
    def world_body_start(self) -> int:
        return int(self.body_world_start_np[0])

    @property
    def viz_half_extents_batched_np(self) -> np.ndarray:
        """Pack partition half extents to length ``num_newton_worlds * max_bodies_per_world`` (mesh extract stride).

        Newton ``body_world_start`` may have more worlds than ``rot_batched_partition`` envs; the ``mb == mx``
        shortcut is unsafe when ``ne > partition_num_envs`` (undersized return vs sim kernels' grid).
        """
        ne = int(len(self.body_world_start_np) - 1)
        mb = int(self.max_bodies_per_world)
        mx = int(self.partition_max_boxes)
        npe = int(self.partition_num_envs)
        hpart = self.partition_half_extents_batched_np
        h = hpart.reshape(npe, mx, 3)
        out = np.zeros((ne * mb, 3), dtype=np.float32)
        for w in range(min(ne, npe)):
            n = int(self.box_counts_per_world[w])
            n = min(n, mx, mb)
            out[w * mb : w * mb + n] = h[w, :n, :]
        return out

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _enforce_frozen_worlds(self) -> None:
        wp.launch(
            enforce_frozen_worlds,
            dim=(_NUM_WORLDS, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                self.world_frozen,
                self.active_mask_gpu,
                self.frozen_body_q,
                self.state_0.body_q,
                self.state_0.body_qd,
            ],
        )

    def step(self) -> None:
        if self.all_settled:
            return
        if self._wall_clock_start is None:
            self._wall_clock_start = time.perf_counter()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()
        self._enforce_frozen_worlds()
        self.frame_count += 1
        self.sim_time += self.frame_dt
        if self.frame_count % self.settle_check_interval == 0:
            self._check_settled()

    def refresh_instant_body_unsettled_flags(self) -> None:
        """Per-body instantaneous lin/ang threshold check (same as ``check_body_stability_lin_ang``).

        Does **not** affect settle counters; use for visualization. ``body_instant_unsettled[gid]==1`` means
        that body currently exceeds |v| or |ω| thresholds for its world.
        """
        self.body_instant_unsettled.zero_()
        wp.launch(
            check_body_stability_lin_ang_per_body_unsettled,
            dim=(_NUM_WORLDS, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                self.world_frozen,
                self.active_mask_gpu,
                self.state_0.body_qd,
                self.world_speed_threshold,
                self.world_angular_threshold,
                self.body_instant_unsettled,
            ],
        )

    def _check_settled(self) -> None:
        self.world_unsettled.zero_()
        wp.launch(
            check_body_stability_lin_ang,
            dim=(_NUM_WORLDS, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                self.world_frozen,
                self.active_mask_gpu,
                self.state_0.body_qd,
                self.world_speed_threshold,
                self.world_angular_threshold,
                self.world_unsettled,
            ],
        )
        wp.synchronize()
        unsettled = self.world_unsettled.numpy()
        newly_frozen = np.zeros(_NUM_WORLDS, dtype=np.int32)
        for w in range(_NUM_WORLDS):
            if self.worlds_settled[w]:
                continue
            if unsettled[w] == 0:
                self.settle_consecutive[w] += 1
            else:
                self.settle_consecutive[w] = 0
            if self.settle_consecutive[w] >= self.settle_steps_required:
                self.worlds_settled[w] = True
                self.world_frozen_np[w] = 1
                newly_frozen[w] = 1
                print(
                    f"  World {w} settled at wall={self._wall_elapsed_s():.3f}s (sim={self.sim_time:.3f}s)"
                )
        if np.any(newly_frozen):
            world_mask = wp.array(np.asarray(newly_frozen, dtype=np.int32), dtype=wp.int32, device=self.model.device)
            wp.launch(
                capture_frozen_body_state,
                dim=(_NUM_WORLDS, self.max_bodies_per_world),
                inputs=[
                    self.model.body_world_start,
                    world_mask,
                    self.state_0.body_q,
                    self.frozen_body_q,
                ],
            )
            self.world_frozen.assign(self.world_frozen_np)
            self._enforce_frozen_worlds()
        if np.all(self.worlds_settled):
            self.all_settled = True
            print(
                f"[STABLE] Settled at wall={self._wall_elapsed_s():.3f}s sim={self.sim_time:.3f}s."
            )
            self._save_snapshot()

    def _wall_elapsed_s(self) -> float:
        if self._wall_clock_start is None:
            return 0.0
        return time.perf_counter() - self._wall_clock_start

    def _save_snapshot(self) -> None:
        if not self.snapshot_path or self.snapshot_saved:
            return
        snapshot_path = Path(self.snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        meta_out = dict(self.snapshot_metadata)
        meta_out["wall_elapsed_s_at_settle"] = float(self._wall_elapsed_s())
        meta_out["sim_time_at_settle"] = float(self.sim_time)
        bq = self.state_0.body_q.numpy()
        bqd = np.zeros_like(self.state_0.body_qd.numpy(), dtype=np.float32)
        np.savez_compressed(
            snapshot_path,
            metadata_json=json.dumps(meta_out),
            sim_time=np.float32(self.sim_time),
            wall_elapsed_s=np.float32(self._wall_elapsed_s()),
            body_q=bq,
            body_qd=bqd,
            body_world=self.model.body_world.numpy(),
            body_world_start=self.body_world_start_np,
            body_half_extents=self.body_half_extents_np,
            body_diag=self.body_diag_np,
            box_counts_per_world=self.box_counts_per_world,
            world_speed_threshold=self.world_speed_threshold.numpy(),
            world_angular_threshold=self.world_angular_threshold.numpy(),
            worlds_settled=self.worlds_settled.astype(np.int32),
        )
        self.snapshot_saved = True
        print(f"[SNAPSHOT] Saved settled scene to {snapshot_path}")


def _run_headless(sim: PsPartitionSim) -> None:
    while not sim.all_settled:
        sim.step()
    if sim.snapshot_path and not sim.snapshot_saved:
        sim._save_snapshot()


def main() -> None:
    wp.init()
    parser = argparse.ArgumentParser(
        description="Single-world rot partition sim with Polyscope (or --headless)."
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device (e.g. cuda:0)")
    parser.add_argument(
        "--solver",
        type=str,
        default=cs.DEFAULT_SOLVER,
        choices=["xpbd", "mujoco"],
        help="Rigid solver (settings from configs_sim).",
    )
    parser.add_argument("--nb", type=int, default=cs.NB, help="Target box count (partition).")
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=cs.SEED,
        metavar="INT",
        help="RNG seed for rot_batched_partition (trivial + discrete_rotation kernels); default: configs_sim.SEED.",
    )
    parser.add_argument(
        "--dims",
        type=float,
        nargs=3,
        default=list(cs.DIMS),
        metavar=("DX", "DY", "DZ"),
        help="Container extent (m).",
    )
    parser.add_argument(
        "--remove-wall",
        type=int,
        nargs="*",
        default=[],
        metavar="IDX",
        help="Wall indices to omit: 0=left -x, 1=right +x, 2=front -y, 3=back +y.",
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default="",
        help="Optional .npz path written when settled (same schema as rot_partition_sim).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No Polyscope; run until settled (useful with --save-snapshot).",
    )
    args = parser.parse_args()
    if args.device:
        wp.set_device(args.device)

    sim = PsPartitionSim(args)
    if args.headless:
        _run_headless(sim)
    else:
        from utils.ps_viewer import run_polyscope_rot_partition

        run_polyscope_rot_partition(sim)


if __name__ == "__main__":
    main()
