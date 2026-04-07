

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer

from configs import configs_sim as cs
from examples.partition_rot_batch_example import rot_batched_partition
from kernels.sim_kernels import (
    capture_frozen_body_state,
    check_body_stability_lin_ang,
    enforce_frozen_worlds,
)
from utils.create_solver import create_solver
from utils.world_info import estimate_rigid_contact_max, wall_definitions_for_dims


@contextmanager
def _perf_span(perf_dict: dict[str, float], key: str):
    t0 = time.perf_counter()
    yield
    perf_dict[key] = time.perf_counter() - t0


def validate_partition_results(out_counts, num_envs, expected_boxes, solver_type):
    if len(out_counts) != num_envs:
        raise ValueError(
            f"Partition returned {len(out_counts)} environments, expected {num_envs}."
        )

    if solver_type != "mujoco":
        return

    counts = [int(c) for c in out_counts]
    unique_counts = sorted(set(counts))
    if len(unique_counts) != 1:
        raise ValueError(
            "MuJoCo solver requires homogeneous worlds, but partition returned "
            f"non-uniform box counts: {unique_counts}"
        )

    actual_boxes = unique_counts[0]
    if actual_boxes != expected_boxes:
        raise ValueError(
            "MuJoCo solver requires each world to contain the requested number of boxes. "
            f"Expected {expected_boxes}, got {actual_boxes}."
        )


class RotPartitionSim:
    def __init__(self, viewer, args):
        self.fps = cs.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = max(1, int(cs.SIM_SUBSTEPS))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.solver_type = args.solver
        self.drop_height = cs.DROP_HEIGHT

        num_envs = int(args.ne)
        n_target = int(args.nb)
        if num_envs < 1:
            raise ValueError(f"--ne must be >= 1, got {num_envs}")
        if n_target < 1:
            raise ValueError(f"--nb must be >= 1, got {n_target}")
        self.num_envs = num_envs
        self.n_target = n_target

        # Settle semantics (see also `step` / `_check_settled`):
        # - `_check_settled` runs every `settle_check_interval` *viewer frames* (`step()` calls).
        #   Each frame advances physics by `sim_substeps` substeps; substeps are not sampled for settle.
        # - `settle_consecutive[w]` counts *consecutive* times a world passed the check (zero unsettled
        #   bodies). It is NOT "consecutive frames" nor "consecutive substeps" — only frames where
        #   the check runs, and only when that evaluation says stable.
        self.settle_steps_required = cs.SETTLE_STEPS
        self.settle_check_interval = cs.SETTLE_CHECK_INTERVAL
        self.frame_count = 0
        self.snapshot_path = args.save_snapshot
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
        print(f"Running batched partition: {num_envs} envs, {n_target} boxes each...")
        self._perf_init: dict[str, float] = {}
        self._perf_runtime: dict[str, float] = {}
        self._perf_report_printed = False
        # True: sync after physics/enforce each frame so perf buckets match GPU wall (slower).
        self._perf_accurate = bool(getattr(args, "perf_accurate", False))
        # True: no CUDA graph; time each substep collide + solver.step with sync (slower, detailed).
        self._perf_substeps = bool(getattr(args, "perf_substeps", False))

        with _perf_span(self._perf_init, "partition"):
            (_, out_counts, _, _, max_boxes, out_centers, out_quats, out_half_extents) = (
                rot_batched_partition(
                    num_envs=num_envs,
                    dims=dims_list,
                    n_target=n_target,
                    min_ratio=cs.MIN_RATIO,
                    shrink=cs.SHRINK,
                    seed=cs.SEED,
                    device=args.device,
                    isotropic=cs.ISOTROPIC,
                )
            )
        validate_partition_results(out_counts, num_envs, n_target, self.solver_type)
        print("Partition done. Building Newton model...")

        with _perf_span(self._perf_init, "model_builder_fill"):
            main_builder = newton.ModelBuilder()
            try:
                newton.solvers.SolverMuJoCo.register_custom_attributes(main_builder)
            except Exception as exc:
                print(f"[warn] Could not register MuJoCo custom attributes on builder: {exc}")

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
            wt = cs.WALL_THICKNESS
            wall_defs = wall_definitions_for_dims(
                dx, dy, dz, float(wt), float(cs.WALL_SCALE), walls_removed=self._walls_removed
            )
            if self._walls_removed:
                print(
                    f"[rot_partition_sim] Walls omitted (indices): {sorted(self._walls_removed)} "
                    f"=> {len(wall_defs)} wall shape(s) in env"
                )
            body_diagonals = []
            body_half_extents = []
            world_speed_thresholds = []

            for env in range(num_envs):
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

                count = int(out_counts[env])
                for i in range(count):
                    flat = env * max_boxes + i
                    c = out_centers[flat]
                    q = out_quats[flat]
                    he = out_half_extents[flat]

                    spawn_pos = wp.vec3(float(c[0]), float(c[1]), float(c[2]) + self.drop_height)
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

        with _perf_span(self._perf_init, "model_finalize"):
            self.model = main_builder.finalize(device=args.device)

        if self.solver_type == "mujoco":
            with _perf_span(self._perf_init, "mujoco_geom_solimp"):
                solimp_list = list(cs.MUJOCO_SOLIMP)
                arr = self.model.mujoco.geom_solimp.numpy()
                arr[:] = np.asarray(solimp_list, dtype=arr.dtype)
            print(f"Applied MuJoCo geom_solimp={solimp_list}")

        with _perf_span(self._perf_init, "collision_pipeline_and_budgets"):
            total_bodies = int(np.sum(out_counts))
            rigid_contact_max = estimate_rigid_contact_max(
                total_bodies=total_bodies,
                contacts_per_body=cs.DEFAULT_CONTACTS_PER_BODY,
                user_override=cs.RIGID_CONTACT_MAX,
                min_rigid_contact_max=cs.MIN_RIGID_CONTACT_MAX,
            )
            self.rigid_contact_max = rigid_contact_max
            self.model.rigid_contact_max = rigid_contact_max

            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="sap",
                rigid_contact_max=rigid_contact_max,
            )
            self.contacts = self.collision_pipeline.contacts()

            nconmax = cs.NCONMAX if cs.NCONMAX > 0 else max(rigid_contact_max // max(num_envs, 1), 100)
            njmax = cs.NJMAX if cs.NJMAX > 0 else nconmax * 3
            self.nconmax = nconmax
            self.njmax = njmax
        print(
            f"Using contact budgets: rigid_contact_max={rigid_contact_max}, "
            f"nconmax={nconmax}, njmax={njmax}"
        )
        with _perf_span(self._perf_init, "solver_create"):
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

        # Rigid-only scenes: body poses come from the finalized model / snapshot; eval_fk is redundant.
        with _perf_span(self._perf_init, "state_and_gpu_arrays"):
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
            self.worlds_settled = np.zeros(num_envs, dtype=bool)
            self.world_frozen_np = np.zeros(num_envs, dtype=np.int32)
            self.world_frozen = wp.zeros(num_envs, dtype=wp.int32, device=args.device)
            self.settle_consecutive = np.zeros(num_envs, dtype=np.int32)
            self.all_settled = False
            self._wall_clock_start = None

        with _perf_span(self._perf_init, "snapshot_metadata_dict"):
            mujoco_solimp_list = [float(x) for x in cs.MUJOCO_SOLIMP]
            self.snapshot_metadata = {
                "snapshot_version": 1,
                "world_count": int(num_envs),
                "dims": [float(v) for v in self.dims_np.tolist()],
                "solver_type": str(self.solver_type),
                "seed": int(cs.SEED),
                "target_boxes_per_world": int(n_target),
                "box_counts_per_world": [int(v) for v in self.box_counts_per_world.tolist()],
                "min_ratio": float(cs.MIN_RATIO),
                "shrink": float(cs.SHRINK),
                "isotropic": bool(cs.ISOTROPIC),
                "fps": int(self.fps),
                "sim_substeps": int(self.sim_substeps),
                "sim_dt": float(self.sim_dt),
                "drop_height": float(self.drop_height),
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
                "rigid_contact_max": int(self.rigid_contact_max),
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
            }

        max_dim = float(np.max(np.asarray(dims_list, dtype=np.float32)))
        with _perf_span(self._perf_init, "viewer_setup"):
            self.viewer.set_model(self.model)
            spacing = max_dim + cs.ENV_SPACING
            self.viewer.set_world_offsets((spacing, spacing, 0.0))

            cols = int(math.ceil(math.sqrt(num_envs)))
            rows = int(math.ceil(num_envs / cols))
            cx = 0.5 * float(cols - 1) * spacing
            cy = 0.5 * float(rows - 1) * spacing
            z_high = max_dim * 5.0 + spacing * 2.0
            self.viewer.set_camera(
                pos=wp.vec3(cx, cy, z_high),
                pitch=-88.0,
                yaw=0.0,
            )

        if self._perf_substeps:
            self.graph = None
            print(
                "[perf] CUDA graph disabled (--perf-substeps): per-substep "
                "collision_pipeline.collide + solver.step timing (GPU wall via sync)."
            )
        else:
            with _perf_span(self._perf_init, "cuda_graph_capture"):
                self._capture_graph()
        print(f"Newton model ready (solver={self.solver_type}). Starting simulation...")

    def _wall_elapsed_s(self) -> float:
        if self._wall_clock_start is None:
            return 0.0
        return time.perf_counter() - self._wall_clock_start

    def _perf_add_runtime(self, key: str, dt: float) -> None:
        self._perf_runtime[key] = self._perf_runtime.get(key, 0.0) + dt

    def _print_perf_report(self) -> None:
        init_total = sum(self._perf_init.values())
        rt = self._perf_runtime
        rt_total = sum(rt.values())
        acc = self._perf_accurate
        sub = self._perf_substeps
        split_note = (
            "per-frame sync after physics/enforce (GPU wall)"
            if acc
            else (
                "fast path: physics_step/enforce are CPU launch only unless --perf-accurate or --perf-substeps"
                if not sub
                else (
                    "--perf-substeps: GPU physics = substep_collide + substep_solver (no separate physics_step row; "
                    "not double-counted in runtime sum)"
                )
            )
        )
        n_sub = int(self.frame_count) * int(self.sim_substeps)
        phys_line = (
            f"  physics_step (capture_launch or full _simulate per viewer frame): {rt.get('physics_step', 0.0):.6f}"
            if not sub
            else "  physics_step: (not accumulated — same GPU wall as substep_collide + substep_solver)"
        )
        lines = [
            "[perf] rot_partition_sim — init (perf_counter wall, s)",
            *(f"  {k}: {v:.6f}" for k, v in sorted(self._perf_init.items())),
            f"  __init__ total: {init_total:.6f}",
            f"[perf] rot_partition_sim — until settle (cumulative wall, s); {split_note}",
            phys_line,
            f"  enforce_frozen_worlds: {rt.get('enforce_frozen_worlds', 0.0):.6f}",
        ]
        if sub:
            lines.extend(
                [
                    f"  substep_collide (CollisionPipeline.collide per substep, incl. sync): {rt.get('substep_collide', 0.0):.6f}",
                    f"  substep_solver (solver.step per substep, incl. sync): {rt.get('substep_solver', 0.0):.6f}",
                    f"  substep_iterations (viewer_frames × sim_substeps): {n_sub}",
                ]
            )
        lines.extend(
            [
                f"  viewer_render (non-Null only; 0 for ViewerNull): {rt.get('viewer_render', 0.0):.6f}",
                f"  snapshot_npz_save: {rt.get('snapshot_npz_save', 0.0):.6f}",
                f"  runtime sum (above): {rt_total:.6f}",
                f"  sim frames (viewer step): {self.frame_count} | sim_substeps/frame: {self.sim_substeps} | wall since sim start: {self._wall_elapsed_s():.6f}s",
            ]
        )
        print("\n".join(lines))

    def _capture_graph(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self._perf_substeps:
                t_c0 = time.perf_counter()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            if self._perf_substeps:
                wp.synchronize()
                self._perf_add_runtime("substep_collide", time.perf_counter() - t_c0)
                t_s0 = time.perf_counter()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            if self._perf_substeps:
                wp.synchronize()
                self._perf_add_runtime("substep_solver", time.perf_counter() - t_s0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _freeze_new_worlds(self, world_mask_np):
        if not np.any(world_mask_np):
            return
        world_mask = wp.array(np.asarray(world_mask_np, dtype=np.int32), dtype=wp.int32, device=self.model.device)
        wp.launch(
            capture_frozen_body_state,
            dim=(self.num_envs, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                world_mask,
                self.state_0.body_q,
                self.frozen_body_q,
            ],
        )
        self.world_frozen.assign(self.world_frozen_np)
        self._enforce_frozen_worlds()

    def _enforce_frozen_worlds(self):
        wp.launch(
            enforce_frozen_worlds,
            dim=(self.num_envs, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                self.world_frozen,
                self.active_mask_gpu,
                self.frozen_body_q,
                self.state_0.body_q,
                self.state_0.body_qd,
            ],
        )

    def step(self):
        """One viewer frame: `sim_substeps` physics substeps, then optional settle check."""
        if self.all_settled:
            if isinstance(self.viewer, newton.viewer.ViewerNull):
                self.viewer.num_frames = self.viewer.frame_count
            return
        if self._wall_clock_start is None:
            self._wall_clock_start = time.perf_counter()
        t0 = time.perf_counter()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        if self._perf_accurate:
            wp.synchronize()
        if not self._perf_substeps:
            self._perf_add_runtime("physics_step", time.perf_counter() - t0)
        t1 = time.perf_counter()
        self._enforce_frozen_worlds()
        if self._perf_accurate:
            wp.synchronize()
        self._perf_add_runtime("enforce_frozen_worlds", time.perf_counter() - t1)
        self.frame_count += 1
        self.sim_time += self.frame_dt
        # Settle sampling: every N frames, not every substep.
        if self.frame_count % self.settle_check_interval == 0:
            self._check_settled()

    def _check_settled(self):
        self.world_unsettled.zero_()
        wp.launch(
            check_body_stability_lin_ang,
            dim=(self.num_envs, self.max_bodies_per_world),
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
        newly_frozen = np.zeros(self.num_envs, dtype=np.int32)
        for w in range(self.num_envs):
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
                    f"  World {w} settled at wall={self._wall_elapsed_s():.3f}s "
                    f"(sim={self.sim_time:.3f}s)"
                )
        self._freeze_new_worlds(newly_frozen)
        if np.all(self.worlds_settled):
            self.all_settled = True
            print(
                f"[STABLE] All {self.num_envs} worlds settled "
                f"at wall={self._wall_elapsed_s():.3f}s "
                f"(sim={self.sim_time:.3f}s). Stopping simulation."
            )
            self._save_snapshot()

    def _save_snapshot(self):
        if not self.snapshot_path or self.snapshot_saved:
            return

        t0 = time.perf_counter()
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
        self._perf_add_runtime("snapshot_npz_save", time.perf_counter() - t0)
        print(f"[SNAPSHOT] Saved settled scene to {snapshot_path}")

    def render(self):
        if isinstance(self.viewer, newton.viewer.ViewerNull):
            # No log_state; only advance ViewerNull frame_count for is_running().
            self.viewer.end_frame()
        else:
            t0 = time.perf_counter()
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()
            self._perf_add_runtime("viewer_render", time.perf_counter() - t0)
        if self.all_settled and not self._perf_report_printed:
            self._print_perf_report()
            self._perf_report_printed = True


def main():
    wp.init()
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default=cs.DEFAULT_SOLVER,
        choices=["xpbd", "mujoco"],
        help="Rigid solver; iteration/contact/settle/partition settings from configs_sim.",
    )
    parser.add_argument(
        "--ne",
        type=int,
        default=cs.NUM_ENVS,
        help="Number of worlds / parallel environments (default: configs_sim.NUM_ENVS).",
    )
    parser.add_argument(
        "--nb",
        type=int,
        default=cs.NB,
        help="Target boxes per world for GPU partition (default: configs_sim.NB).",
    )
    parser.add_argument(
        "--dims",
        type=float,
        nargs=3,
        default=list(cs.DIMS),
        metavar=("DX", "DY", "DZ"),
        help="World container extent in meters (x, y, z); default: configs_sim.DIMS.",
    )
    parser.add_argument(
        "--remove-wall",
        type=int,
        nargs="*",
        default=[],
        metavar="IDX",
        help=(
            "Omit static container walls by index (default: all four). "
            "0=left (-x), 1=right (+x), 2=front (-y), 3=back (+y). "
            "Example: --remove-wall 0  => three walls; --remove-wall 0 1 2 3  => no walls."
        ),
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default="",
        help="Optional .npz path to save once all worlds have settled",
    )
    parser.add_argument("--vis", action="store_true", help="Open GL viewer; default is ViewerNull (no window).")
    parser.add_argument(
        "--perf-accurate",
        action="store_true",
        help=(
            "After each physics/enforce, wp.synchronize() so frame-level perf matches GPU wall time (slower)."
        ),
    )
    parser.add_argument(
        "--perf-substeps",
        action="store_true",
        help=(
            "Disable CUDA graph and accumulate GPU wall time per simulation substep: "
            "CollisionPipeline.collide vs solver.step (each followed by wp.synchronize()). "
            "Slower; use for profiling contact vs integrator cost."
        ),
    )

    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)
    if args.vis:
        viewer = newton.viewer.ViewerGL(headless=args.headless)
    else:
        viewer = newton.viewer.ViewerNull(num_frames=10**9)

    example = RotPartitionSim(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
