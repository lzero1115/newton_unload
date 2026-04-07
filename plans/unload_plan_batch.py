# Batch unload: load snapshot world 0 from .npz (metadata_json matches rot_partition_sim / batch snapshot schema),
# replicate N identical Newton worlds, remove boxes per world with pluggable RemovalPlanner.
# Verify: N full frames then stability check per world; removal loop uses per-world cooldown / settle counts.

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer

from configs import configs_sim as cs
from configs.unload_configs import (
    DEFAULT_SETTLE_COOLDOWN_FRAMES,
    ENV_SPACING,
    INTER_STEADY_METRIC_ROT_SCALE,
    REMOVED_BODY_Z,
)
from kernels.sim_kernels import (
    check_body_stability_lin_ang,
    enforce_removed_bodies,
    inter_steady_metric_weighted_sum_per_world_masked,
    zero_world_active_velocities_masked,
)
from unload_planner.factory import make_planner
from utils.create_solver import create_solver
from utils.world_info import (
    compute_world_speed_stats_numpy,
    estimate_rigid_contact_max,
    load_snapshot_metadata,
    wall_definitions_for_dims,
)

_LOG = "[unload_plan_batch]"


@contextmanager
def _perf_span(perf_dict: dict[str, float], key: str):
    t0 = time.perf_counter()
    yield
    perf_dict[key] = time.perf_counter() - t0


class UnloadPlanBatchSim:
    """Replicate snapshot world0 N times; each world removes boxes independently."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device
        self.snapshot_path = Path(args.snapshot)
        self.user_world_count = int(args.world_count)
        if self.user_world_count < 1:
            raise ValueError("--world-count must be >= 1.")
        self.verify_steps = int(args.verify_steps)
        self.verify_threshold_scale = float(args.verify_threshold_scale)
        self.inter_steady_metric_total = np.zeros(self.user_world_count, dtype=np.float64)
        self._inter_steady_metric_step = np.zeros(self.user_world_count, dtype=np.int32)

        self._perf_init: dict[str, float] = {}
        self._perf_runtime: dict[str, float] = {}
        self._perf_report_printed = False

        with _perf_span(self._perf_init, "snapshot_io"):
            snapshot = np.load(self.snapshot_path, allow_pickle=False)
            metadata = load_snapshot_metadata(snapshot)
            self.metadata = metadata

            orig_world_count = int(metadata["world_count"])
            if orig_world_count < 1:
                raise ValueError("Snapshot world_count must be >= 1.")

            self.fps = int(metadata.get("fps", cs.FPS))
            self.frame_dt = 1.0 / self.fps
            self.sim_substeps = int(metadata.get("sim_substeps", cs.SIM_SUBSTEPS))
            self.sim_dt = float(metadata.get("sim_dt", self.frame_dt / self.sim_substeps))

            self.solver_type = str(metadata.get("solver_type", cs.DEFAULT_SOLVER))
            if self.solver_type not in ("xpbd", "mujoco"):
                raise ValueError(f"Unsupported solver in snapshot metadata: {self.solver_type}")

            self.settle_steps_required = int(metadata.get("settle_steps", cs.SETTLE_STEPS))
            self.settle_check_interval = int(metadata.get("settle_check_interval", cs.SETTLE_CHECK_INTERVAL))
            self.settle_cooldown_frames = int(
                args.settle_cooldown_frames
                if args.settle_cooldown_frames >= 0
                else DEFAULT_SETTLE_COOLDOWN_FRAMES
            )

            self.dims_np = np.asarray(metadata["dims"], dtype=np.float32)
            body_q_full = np.asarray(snapshot["body_q"], dtype=np.float32)
            body_qd_full = np.asarray(snapshot["body_qd"], dtype=np.float32)
            body_he_full = np.asarray(snapshot["body_half_extents"], dtype=np.float32)
            body_diag_full = np.asarray(snapshot["body_diag"], dtype=np.float32)
            body_world_start_full = np.asarray(snapshot["body_world_start"], dtype=np.int32)
            w_start = int(body_world_start_full[0])
            w_end = int(body_world_start_full[1])
            n_boxes = w_end - w_start
            if n_boxes <= 0:
                raise ValueError("World 0 has no bodies in snapshot.")

            self.body_q_np = body_q_full[w_start:w_end].copy()
            self.body_qd_np = body_qd_full[w_start:w_end].copy()
            self.body_half_extents_np = body_he_full[w_start:w_end].copy()
            self.body_diag_np = body_diag_full[w_start:w_end].copy()

            wthr = np.asarray(snapshot["world_speed_threshold"], dtype=np.float32)
            if wthr.size <= 0:
                raise ValueError("snapshot missing world_speed_threshold")
            self.world_speed_threshold_scalar = float(wthr[0])

            if "world_angular_threshold" in snapshot.files:
                wthr_a = np.asarray(snapshot["world_angular_threshold"], dtype=np.float32).flatten()
                self.world_angular_threshold_scalar = float(wthr_a[0]) if wthr_a.size > 0 else float(
                    metadata.get("settle_angular_speed_rad", cs.SETTLE_ANGULAR_SPEED_RAD)
                )
            else:
                self.world_angular_threshold_scalar = float(
                    metadata.get("settle_angular_speed_rad", cs.SETTLE_ANGULAR_SPEED_RAD)
                )

            # New snapshots export `settle_linear_speed_mps` / `settle_angular_speed_rad`; override npz if present.
            self.world_speed_threshold_scalar = float(
                metadata.get("settle_linear_speed_mps", self.world_speed_threshold_scalar)
            )
            self.world_angular_threshold_scalar = float(
                metadata.get("settle_angular_speed_rad", self.world_angular_threshold_scalar)
            )

            self.sim_time = float(np.asarray(snapshot["sim_time"]).item())

        print(
            f"{_LOG} Loading {self.snapshot_path}: replicate world0 "
            f"({n_boxes} boxes) × {self.user_world_count} worlds, solver={self.solver_type}, "
            f"seed={args.seed}, planner={args.removal_planner}"
        )

        with _perf_span(self._perf_init, "model_build"):
            self._build_model_and_state(metadata, n_boxes)
        with _perf_span(self._perf_init, "removal_arrays"):
            self._setup_removal_arrays(n_boxes)

        self.planners = [
            make_planner(args.removal_planner, np.random.default_rng(int(args.seed) + w), self)
            for w in range(self.world_count)
        ]

        self.frame_count = 0
        self.world_cooldown_remaining = np.zeros(self.world_count, dtype=np.int32)
        self.world_settle_consecutive = np.zeros(self.world_count, dtype=np.int32)
        self.world_first_removal_done = np.zeros(self.world_count, dtype=bool)
        self.world_done = np.zeros(self.world_count, dtype=bool)
        self._sim_wall_t0: float | None = None
        self._sim_wall_time_reported = False

        with _perf_span(self._perf_init, "verify_phase"):
            self._run_verify_phase()

        print(
            f"{_LOG} Verify done. Removal: settle_check_interval={self.settle_check_interval}, "
            f"settle_steps_required={self.settle_steps_required}, cooldown_frames={self.settle_cooldown_frames}; "
            f"inter_steady_performance per world (rot_scale={INTER_STEADY_METRIC_ROT_SCALE})."
        )

        with _perf_span(self._perf_init, "viewer_setup"):
            self._setup_viewer()

    def _solver_kwargs_from_metadata(self):
        m = self.metadata
        return {
            "xpbd_iterations": int(m.get("xpbd_iterations", cs.XPBD_ITERATIONS)),
            "xpbd_contact_relaxation": float(m.get("xpbd_contact_relaxation", cs.XPBD_CONTACT_RELAXATION)),
            "xpbd_angular_damping": float(
                m.get("xpbd_angular_damping", m.get("angular_damping", cs.ANGULAR_DAMPING))
            ),
            "xpbd_enable_restitution": bool(m.get("xpbd_enable_restitution", cs.XPBD_ENABLE_RESTITUTION)),
            "mujoco_iterations": int(m.get("mujoco_iterations", cs.MUJOCO_ITERATIONS)),
            "mujoco_ls_iterations": int(m.get("mujoco_ls_iterations", cs.MUJOCO_LS_ITERATIONS)),
            "mujoco_solver": str(m.get("mujoco_solver", cs.MUJOCO_SOLVER)),
            "mujoco_integrator": str(m.get("mujoco_integrator", cs.MUJOCO_INTEGRATOR)),
            "mujoco_cone": str(m.get("mujoco_cone", cs.MUJOCO_CONE)),
            "mujoco_impratio": float(m.get("mujoco_impratio", cs.MUJOCO_IMPRATIO)),
            "mujoco_tolerance": float(m.get("mujoco_tolerance", cs.MUJOCO_TOLERANCE)),
            "mujoco_ls_tolerance": float(m.get("mujoco_ls_tolerance", cs.MUJOCO_LS_TOLERANCE)),
            "mujoco_update_data_interval": int(m.get("mujoco_update_data_interval", cs.MUJOCO_UPDATE_DATA_INTERVAL)),
            "mujoco_use_contacts": bool(m.get("mujoco_use_contacts", cs.MUJOCO_USE_CONTACTS)),
        }

    def _build_model_and_state(self, metadata, n: int):
        rigid_gap = float(metadata.get("rigid_gap", cs.DEFAULT_RIGID_GAP))
        ke = float(metadata.get("mujoco_contact_ke", cs.MUJOCO_CONTACT_KE))
        kd_default = float(metadata.get("mujoco_contact_kd", cs.MUJOCO_CONTACT_KD))

        main_builder = newton.ModelBuilder()
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(main_builder)
        except Exception as exc:
            print(f"[warn] Could not register MuJoCo custom attributes: {exc}")

        main_builder.rigid_gap = rigid_gap
        main_builder.default_shape_cfg.gap = rigid_gap
        main_builder.default_shape_cfg.ke = ke
        main_builder.default_shape_cfg.kd = kd_default

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(metadata.get("ground_mu", cs.GROUND_MU)),
            kd=float(metadata.get("ground_kd", cs.GROUND_KD)),
            mu_torsional=float(metadata.get("ground_mu_torsional", cs.GROUND_MU_TORSIONAL)),
            mu_rolling=float(metadata.get("ground_mu_rolling", cs.GROUND_MU_ROLLING)),
            gap=rigid_gap,
            ke=ke,
        )
        main_builder.add_shape_plane(
            body=-1,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=ground_cfg,
        )

        box_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(metadata.get("box_mu", cs.BOX_MU)),
            kd=float(metadata.get("box_kd", cs.BOX_KD)),
            mu_torsional=float(metadata.get("box_mu_torsional", cs.BOX_MU_TORSIONAL)),
            mu_rolling=float(metadata.get("box_mu_rolling", cs.BOX_MU_ROLLING)),
            gap=rigid_gap,
            ke=ke,
        )
        wall_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(metadata.get("wall_mu", cs.WALL_MU)),
            kd=float(metadata.get("wall_kd", cs.WALL_KD)),
            mu_torsional=float(metadata.get("wall_mu_torsional", cs.WALL_MU_TORSIONAL)),
            mu_rolling=float(metadata.get("wall_mu_rolling", cs.WALL_MU_ROLLING)),
            gap=rigid_gap,
            ke=ke,
        )

        dx, dy, dz = [float(v) for v in self.dims_np.tolist()]
        wall_thickness = float(metadata.get("wall_thickness", cs.WALL_THICKNESS))
        wall_scale = float(metadata.get("wall_scale", cs.WALL_SCALE))
        wr = metadata.get("walls_removed", [])
        walls_removed = {int(x) for x in wr} if wr else set()
        wall_defs = wall_definitions_for_dims(
            dx, dy, dz, wall_thickness, wall_scale, walls_removed=walls_removed
        )

        # Single template world, then ModelBuilder.replicate (same as parallel_boxes / basic URDF examples).
        # spacing=(0,0,0): keep all copies at origin in physics; viewer set_world_offsets() separates visually.
        env_builder = newton.ModelBuilder()
        env_builder.rigid_gap = rigid_gap
        env_builder.default_shape_cfg.gap = rigid_gap
        env_builder.default_shape_cfg.ke = ke
        env_builder.default_shape_cfg.kd = kd_default

        for pos, whx, why, whz_w in wall_defs:
            env_builder.add_shape_box(
                body=-1,
                xform=wp.transform(pos, wp.quat_identity()),
                hx=whx,
                hy=why,
                hz=whz_w,
                cfg=wall_shape_cfg,
            )

        for i in range(n):
            pose = self.body_q_np[i]
            he = self.body_half_extents_np[i]
            body = env_builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
                    q=wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
                )
            )
            env_builder.add_shape_box(
                body,
                hx=float(he[0]),
                hy=float(he[1]),
                hz=float(he[2]),
                cfg=box_shape_cfg,
            )

        main_builder.replicate(env_builder, world_count=self.user_world_count, spacing=(0.0, 0.0, 0.0))

        self.model = main_builder.finalize(device=self.device)
        self.world_count = self.user_world_count

        if self.solver_type == "mujoco":
            solimp_src = metadata.get("mujoco_solimp", list(cs.MUJOCO_SOLIMP))
            solimp = [float(x) for x in solimp_src]
            arr = self.model.mujoco.geom_solimp.numpy()
            arr[:] = np.asarray(solimp, dtype=arr.dtype)
            print(f"{_LOG} Applied MuJoCo geom_solimp={solimp}")

        contacts_pb = int(metadata.get("contacts_per_body", cs.DEFAULT_CONTACTS_PER_BODY))
        total_box_bodies = n * self.world_count
        saved_rigid_contact_max = int(metadata.get("rigid_contact_max", 0))
        rigid_contact_max = estimate_rigid_contact_max(
            total_bodies=total_box_bodies,
            contacts_per_body=contacts_pb,
            user_override=saved_rigid_contact_max,
            min_rigid_contact_max=cs.MIN_RIGID_CONTACT_MAX,
        )
        self.model.rigid_contact_max = rigid_contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=rigid_contact_max,
        )
        self.contacts = self.collision_pipeline.contacts()

        sk = self._solver_kwargs_from_metadata()
        saved_nconmax = int(metadata.get("nconmax", cs.NCONMAX))
        saved_njmax = int(metadata.get("njmax", cs.NJMAX))
        nconmax = saved_nconmax if saved_nconmax > 0 else max(rigid_contact_max // max(self.world_count, 1), 100)
        njmax = saved_njmax if saved_njmax > 0 else nconmax * 3

        self.solver = create_solver(
            self.model,
            self.solver_type,
            nconmax=nconmax,
            njmax=njmax,
            **sk,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Snapshot overwrites body state below; rigid-only — no eval_fk.

        self.body_world_start_np = self.model.body_world_start.numpy()
        for w in range(self.world_count):
            w0 = int(self.body_world_start_np[w])
            w1 = int(self.body_world_start_np[w + 1])
            self.state_0.body_q[w0:w1].assign(self.body_q_np)
            self.state_0.body_qd[w0:w1].assign(self.body_qd_np)
        self.state_1.assign(self.state_0)

        body_counts = np.diff(self.body_world_start_np[: self.world_count + 1])
        self.max_bodies_per_world = int(np.max(body_counts)) if len(body_counts) else n

        self.body_diag = wp.array(
            np.tile(self.body_diag_np, self.world_count), dtype=wp.float32, device=self.device
        )
        thr_one = np.array([self.world_speed_threshold_scalar], dtype=np.float32)
        self._world_speed_threshold_base_np = np.tile(thr_one, self.world_count)
        self.world_speed_threshold = wp.array(self._world_speed_threshold_base_np, dtype=wp.float32, device=self.device)
        ang_one = np.array([self.world_angular_threshold_scalar], dtype=np.float32)
        self._world_angular_threshold_base_np = np.tile(ang_one, self.world_count)
        self.world_angular_threshold = wp.array(self._world_angular_threshold_base_np, dtype=wp.float32, device=self.device)
        self.world_frozen = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.world_unsettled = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)

        he_full = np.zeros((self.model.body_count, 3), dtype=np.float32)
        for w in range(self.world_count):
            w0 = int(self.body_world_start_np[w])
            w1 = int(self.body_world_start_np[w + 1])
            he_full[w0:w1] = self.body_half_extents_np
        self.body_half_extents_gpu = wp.array(he_full, dtype=wp.vec3, device=self.device)

        print(
            f"{_LOG} Model ready (replicate): rigid_contact_max={rigid_contact_max}, "
            f"n_boxes_per_world={n}, worlds={self.world_count}, max_bodies_per_world={self.max_bodies_per_world}"
        )

    def _setup_removal_arrays(self, n_boxes: int):
        self.body_count = self.model.body_count
        self.shape_body_np = self.model.shape_body.numpy()
        self.shape_indices_by_body = self._build_shape_index_map()
        self.active_indices_by_world = []
        for w in range(self.world_count):
            w0 = int(self.body_world_start_np[w])
            self.active_indices_by_world.append(list(range(w0, w0 + n_boxes)))
        self.active_mask = np.ones(self.body_count, dtype=np.int32)
        self.active_mask_gpu = wp.array(self.active_mask, dtype=wp.int32, device=self.device)
        self.frozen_body_q = wp.clone(self.state_0.body_q)
        self.body_q_prev = wp.clone(self.state_0.body_q)
        self.metric_sum_per_world = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.world_mask_gpu = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)

    def _build_shape_index_map(self):
        shape_indices_by_body = {body: [] for body in range(self.body_count)}
        for shape_index, body_index in enumerate(self.shape_body_np):
            body_index = int(body_index)
            if body_index >= 0:
                shape_indices_by_body[body_index].append(shape_index)
        return shape_indices_by_body

    def _sync_active_mask(self):
        self.active_mask_gpu = wp.array(self.active_mask, dtype=wp.int32, device=self.device)

    def _set_frozen_graveyard_pose(self, body_index: int):
        buf = self.frozen_body_q.numpy().copy()
        ref = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, REMOVED_BODY_Z), wp.quat_identity())],
            dtype=wp.transform,
            device="cpu",
        )
        buf[body_index] = ref.numpy()[0]
        self.frozen_body_q.assign(buf)

    def _disable_body_collisions(self, body_index: int):
        col_group = self.model.shape_collision_group.numpy().copy()
        for shape_index in self.shape_indices_by_body[body_index]:
            col_group[shape_index] = 0
        self.model.shape_collision_group.assign(col_group)

    def _enforce_removed_bodies(self):
        wp.launch(
            enforce_removed_bodies,
            dim=(self.world_count, self.max_bodies_per_world),
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.active_mask_gpu,
                self.frozen_body_q,
                self.model.body_world_start,
            ],
        )

    def _launch_zero_velocities_masked(self, mask_np: np.ndarray):
        """Single kernel launch: mask_np[w]!=0 => zero active velocities in world w."""
        if not np.any(mask_np):
            return
        self.world_mask_gpu.assign(np.asarray(mask_np, dtype=np.int32))
        wp.launch(
            zero_world_active_velocities_masked,
            dim=(self.world_count, self.max_bodies_per_world),
            inputs=[
                self.state_0.body_qd,
                self.active_mask_gpu,
                self.model.body_world_start,
                self.world_mask_gpu,
            ],
        )

    def _simulate_frame(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._enforce_removed_bodies()

    def _launch_stability_check(self) -> np.ndarray:
        self.world_unsettled.zero_()
        wp.launch(
            check_body_stability_lin_ang,
            dim=(self.world_count, self.max_bodies_per_world),
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
        return self.world_unsettled.numpy().copy()

    def _mark_sim_wall_start(self):
        if self._sim_wall_t0 is None:
            self._sim_wall_t0 = time.perf_counter()

    def _wall_elapsed_s(self) -> float:
        if self._sim_wall_t0 is None:
            return 0.0
        return time.perf_counter() - self._sim_wall_t0

    def _perf_add_runtime(self, key: str, dt: float) -> None:
        self._perf_runtime[key] = self._perf_runtime.get(key, 0.0) + dt

    def _print_perf_report(self) -> None:
        init_total = sum(self._perf_init.values())
        rt = self._perf_runtime
        rt_total = sum(rt.values())
        lines = [
            "[perf] unload_plan_batch — init (perf_counter wall, s)",
            *(f"  {k}: {v:.6f}" for k, v in sorted(self._perf_init.items())),
            f"  __init__ total: {init_total:.6f}",
            "[perf] unload_plan_batch — runtime (cumulative wall, s); "
            "physics_frame/stability_check are CPU-side around GPU launches; "
            "stability_check includes readback via numpy.",
            f"  physics_frame: {rt.get('physics_frame', 0.0):.6f}",
            f"  stability_check: {rt.get('stability_check', 0.0):.6f}",
            f"  viewer_render (non-Null only; 0 for ViewerNull): {rt.get('viewer_render', 0.0):.6f}",
            f"  runtime sum (above): {rt_total:.6f}",
            f"  sim frames (viewer step): {self.frame_count} | sim_substeps/frame: {self.sim_substeps} | "
            f"wall since sim start: {self._wall_elapsed_s():.6f}s",
        ]
        print("\n".join(lines))

    def _report_sim_wall_time_once(self):
        if self._sim_wall_time_reported or self._sim_wall_t0 is None:
            return
        self._sim_wall_time_reported = True
        elapsed = time.perf_counter() - self._sim_wall_t0
        print(
            f"{_LOG} Simulation wall time (perf_counter): {elapsed:.4f}s | "
            f"sim_time={self.sim_time:.4f}s"
        )

    def _maybe_report_sim_wall_finish(self):
        if not bool(np.all(self.world_done)):
            return
        self._report_sim_wall_time_once()

    def _run_verify_phase(self):
        self._mark_sim_wall_start()
        print(
            f"{_LOG} Verify: {self.verify_steps} full frames, then stability check per world..."
        )
        for _ in range(self.verify_steps):
            self._simulate_frame()
            self.sim_time += self.frame_dt

        thr_lin_np = self._world_speed_threshold_base_np * float(self.verify_threshold_scale)
        thr_ang_np = self._world_angular_threshold_base_np * float(self.verify_threshold_scale)
        if self.verify_threshold_scale != 1.0:
            self.world_speed_threshold.assign(thr_lin_np)
            self.world_angular_threshold.assign(thr_ang_np)
        unsettled = self._launch_stability_check()
        if self.verify_threshold_scale != 1.0:
            self.world_speed_threshold.assign(self._world_speed_threshold_base_np)
            self.world_angular_threshold.assign(self._world_angular_threshold_base_np)

        body_qd_np = self.state_0.body_qd.numpy()
        body_diag_np = self.body_diag.numpy()
        any_bad = False
        for w in range(self.world_count):
            _, max_lin, max_ang, max_equiv = compute_world_speed_stats_numpy(
                self.body_world_start_np,
                body_qd_np,
                body_diag_np,
                thr_lin_np,
                w,
            )
            thr_lin = float(thr_lin_np[w])
            thr_ang = float(thr_ang_np[w])
            u = int(unsettled[w])
            thr_note = ""
            if self.verify_threshold_scale != 1.0:
                thr_note = (
                    f" (base lin={self.world_speed_threshold_scalar:.4f} m/s, base ang={self.world_angular_threshold_scalar:.4f} rad/s "
                    f"× verify_threshold_scale={self.verify_threshold_scale:g})"
                )
            print(
                f"{_LOG} world {w}: unsettled_count={u}, lin_thr={thr_lin:.4f} m/s, ang_thr={thr_ang:.4f} rad/s{thr_note}, "
                f"max_lin={max_lin:.4f} m/s, max_ang={max_ang:.4f} rad/s, max_equiv={max_equiv:.4f} m/s"
            )
            if u > 0:
                any_bad = True

        if any_bad:
            print(
                "initial scene unstable on at least one world vs snapshot threshold "
                "(replay drift is common; try --verify-threshold-scale 1.15 or more --verify-steps)."
            )
            try:
                self.viewer.close()
            except Exception:
                pass
            sys.exit(1)

        self.body_q_prev.assign(self.state_0.body_q)

    def _on_inter_steady_metric_batch(self, world_ids: list[int]):
        """Volume-weighted inter-steady performance vs previous steady pose; updates body_q_prev per world."""
        if not world_ids:
            return
        m = np.zeros(self.world_count, dtype=np.int32)
        for w in world_ids:
            m[w] = 1
        self.world_mask_gpu.assign(m)
        self.metric_sum_per_world.zero_()
        wp.launch(
            inter_steady_metric_weighted_sum_per_world_masked,
            dim=(self.world_count, self.max_bodies_per_world),
            inputs=[
                self.model.body_world_start,
                self.active_mask_gpu,
                self.body_q_prev,
                self.state_0.body_q,
                self.body_half_extents_gpu,
                self.body_diag,
                float(INTER_STEADY_METRIC_ROT_SCALE),
                self.world_mask_gpu,
                self.metric_sum_per_world,
            ],
        )
        deltas = self.metric_sum_per_world.numpy()
        for w in world_ids:
            delta = float(deltas[w])
            self._inter_steady_metric_step[w] += 1
            self.inter_steady_metric_total[w] += delta
            print(
                f"{_LOG} world {w} inter_steady_performance step "
                f"{int(self._inter_steady_metric_step[w])}: delta={delta:.6f} "
                f"cumulative={self.inter_steady_metric_total[w]:.6f}"
            )
        for w in world_ids:
            self._copy_body_q_prev_segment_from_state(w)

    def _copy_body_q_prev_segment_from_state(self, world_id: int):
        w0 = int(self.body_world_start_np[world_id])
        w1 = int(self.body_world_start_np[world_id + 1])
        buf = self.body_q_prev.numpy()
        buf[w0:w1] = self.state_0.body_q.numpy()[w0:w1]
        self.body_q_prev.assign(buf)

    def _remove_one_box(self, world_id: int, body_index: int):
        lst = self.active_indices_by_world[world_id]
        lst.remove(body_index)
        self.active_mask[body_index] = 0
        self._disable_body_collisions(body_index)
        self._set_frozen_graveyard_pose(body_index)
        self._sync_active_mask()
        self._enforce_removed_bodies()
        print(
            f"{_LOG} world {world_id}: removed body {body_index}, "
            f"{len(lst)} box(es) remain."
        )

    def _setup_viewer(self):
        self.viewer.set_model(self.model)
        spacing = float(np.max(self.dims_np) + ENV_SPACING)
        self.viewer.set_world_offsets((spacing, spacing, 0.0))
        dx, dy = float(self.dims_np[0]), float(self.dims_np[1])
        max_dim = float(np.max(self.dims_np))
        cx = 0.5 * dx
        cy = 0.5 * dy
        z_high = max_dim * 5.0 + spacing * 2.0
        self.viewer.set_camera(
            pos=wp.vec3(cx, cy, z_high),
            pitch=-88.0,
            yaw=0.0,
        )

    def step(self):
        if bool(np.all(self.world_done)):
            # ViewerNull stops via frame cap; shrink cap so run() exits once sim is finished.
            if isinstance(self.viewer, newton.viewer.ViewerNull):
                self.viewer.num_frames = self.viewer.frame_count
            return

        t_phys = time.perf_counter()
        self._simulate_frame()
        self._perf_add_runtime("physics_frame", time.perf_counter() - t_phys)
        self.sim_time += self.frame_dt
        self.frame_count += 1

        need_settle_check = self.frame_count % self.settle_check_interval == 0
        if need_settle_check:
            t_stab = time.perf_counter()
            unsettled = self._launch_stability_check()
            self._perf_add_runtime("stability_check", time.perf_counter() - t_stab)
        else:
            unsettled = None

        cooldown_before = np.array(self.world_cooldown_remaining, dtype=np.int32)
        for w in range(self.world_count):
            if self.world_done[w]:
                continue
            if self.world_cooldown_remaining[w] > 0:
                self.world_cooldown_remaining[w] -= 1

        first_removal_worlds: list[int] = []
        for w in range(self.world_count):
            if self.world_done[w]:
                continue
            if cooldown_before[w] > 0:
                continue
            if self.world_first_removal_done[w]:
                continue
            if len(self.active_indices_by_world[w]) <= 1:
                self.world_done[w] = True
                print(f"{_LOG} world {w}: only one box left; nothing to remove.")
                continue
            first_removal_worlds.append(w)

        first_removal_this_frame: set[int] = set()
        if first_removal_worlds:
            zmask = np.zeros(self.world_count, dtype=np.int32)
            for w in first_removal_worlds:
                zmask[w] = 1
            self._launch_zero_velocities_masked(zmask)
            for w in first_removal_worlds:
                chosen = self.planners[w].select_body_to_remove(self.active_indices_by_world[w])
                self._remove_one_box(w, chosen)
                self.world_first_removal_done[w] = True
                self.world_settle_consecutive[w] = 0
                self.world_cooldown_remaining[w] = self.settle_cooldown_frames
                first_removal_this_frame.add(w)
                if len(self.active_indices_by_world[w]) <= 1:
                    self.world_done[w] = True
                    print(
                        f"{_LOG} world {w}: inter_steady_performance total="
                        f"{self.inter_steady_metric_total[w]:.6f} (sim_t={self.sim_time:.4f}s)"
                    )
                    print(f"{_LOG} world {w}: done (single box remains).")

        if need_settle_check:
            ready_worlds: list[int] = []
            for w in range(self.world_count):
                if self.world_done[w]:
                    continue
                if cooldown_before[w] > 0:
                    continue
                # Match old single-loop semantics: first removal uses `continue`, so no
                # settle check on the same frame as that removal for this world.
                if w in first_removal_this_frame:
                    continue
                if not self.world_first_removal_done[w]:
                    continue
                if unsettled is not None and unsettled[w] > 0:
                    self.world_settle_consecutive[w] = 0
                    continue
                self.world_settle_consecutive[w] += 1
                if self.world_settle_consecutive[w] < self.settle_steps_required:
                    continue
                self.world_settle_consecutive[w] = 0
                ready_worlds.append(w)

            if ready_worlds:
                self._on_inter_steady_metric_batch(ready_worlds)

                for w in ready_worlds:
                    if len(self.active_indices_by_world[w]) <= 1:
                        self.world_done[w] = True
                        print(
                            f"{_LOG} world {w}: inter_steady_performance total="
                            f"{self.inter_steady_metric_total[w]:.6f} (sim_t={self.sim_time:.4f}s)"
                        )
                        print(
                            f"{_LOG} world {w}: done (single box) at sim_t={self.sim_time:.4f}s."
                        )

                removal_worlds = [
                    w
                    for w in ready_worlds
                    if not self.world_done[w] and len(self.active_indices_by_world[w]) > 1
                ]
                if removal_worlds:
                    zmask = np.zeros(self.world_count, dtype=np.int32)
                    for w in removal_worlds:
                        zmask[w] = 1
                    self._launch_zero_velocities_masked(zmask)
                    for w in removal_worlds:
                        chosen = self.planners[w].select_body_to_remove(self.active_indices_by_world[w])
                        self._remove_one_box(w, chosen)
                        self.world_settle_consecutive[w] = 0
                        self.world_cooldown_remaining[w] = self.settle_cooldown_frames
                        if len(self.active_indices_by_world[w]) <= 1:
                            self.world_done[w] = True
                            print(
                                f"{_LOG} world {w}: inter_steady_performance total="
                                f"{self.inter_steady_metric_total[w]:.6f} (sim_t={self.sim_time:.4f}s)"
                            )
                            print(
                                f"{_LOG} world {w}: done (single box) at sim_t={self.sim_time:.4f}s."
                            )

        self._maybe_report_sim_wall_finish()

    def render(self):
        if isinstance(self.viewer, newton.viewer.ViewerNull):
            self.viewer.end_frame()
        else:
            t0 = time.perf_counter()
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()
            self._perf_add_runtime("viewer_render", time.perf_counter() - t0)
        if bool(np.all(self.world_done)) and not self._perf_report_printed:
            self._print_perf_report()
            self._perf_report_printed = True


def main():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to .npz snapshot (e.g. from rot_partition_sim.py --save-snapshot)",
    )
    parser.add_argument(
        "--world-count",
        "--ne",
        type=int,
        default=2,
        help="Number of replicated worlds (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-world seed is seed+world_id")
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=10,
        help="Full frames to simulate before one stability check per world",
    )
    parser.add_argument(
        "--verify-threshold-scale",
        type=float,
        default=1.0,
        help="Multiply snapshot speed threshold only for the initial verify check",
    )
    parser.add_argument(
        "--removal_planner",
        type=str,
        default="random",
        choices=["random", "height"],
        help="Policy for picking the next box to remove per world",
    )
    parser.add_argument(
        "--settle-cooldown-frames",
        type=int,
        default=-1,
        help=(
            "Frames to skip per-world removal/settle logic after each removal; "
            f"-1 = {DEFAULT_SETTLE_COOLDOWN_FRAMES} (unload-only, not in snapshot)"
        ),
    )
    parser.add_argument("--vis", action="store_true", help="Open GL viewer; default is ViewerNull (no window).")

    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)
    if args.vis:
        viewer = newton.viewer.ViewerGL(headless=args.headless)
    else:
        viewer = newton.viewer.ViewerNull(num_frames=10**9)

    example = UnloadPlanBatchSim(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()


# Backward-compatible name for imports from older scripts.
UnloadPlanMultiSim = UnloadPlanBatchSim
