# Snapshot load + replicated unload scene + the same initial verify as unload_plan_batch,
# without planners, removal loop, perf, or viewer setup (ViewerNull only for verify failure path).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

from configs import configs_sim as cs
from configs.unload_configs import DEFAULT_SETTLE_COOLDOWN_FRAMES
from kernels.sim_kernels import check_body_stability_lin_ang, enforce_removed_bodies
from utils.create_solver import create_solver
from utils.world_info import (
    compute_world_speed_stats_numpy,
    estimate_rigid_contact_max,
    load_snapshot_metadata,
    wall_definitions_for_dims,
)

_LOG = "[initial_verify_snapshot]"


class InitialVerifySnapshot:
    """Same physics + initial stability check as ``UnloadPlanBatchSim``, minimal surface area."""

    def __init__(self, args):
        self.viewer = newton.viewer.ViewerNull(num_frames=1)
        self.device = args.device
        self.snapshot_path = Path(args.snapshot)
        self.user_world_count = int(args.world_count)
        if self.user_world_count < 1:
            raise ValueError("--world-count must be >= 1.")
        self.verify_steps = int(args.verify_steps)
        self.verify_threshold_scale = float(args.verify_threshold_scale)

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

        self.world_speed_threshold_scalar = float(
            metadata.get("settle_linear_speed_mps", self.world_speed_threshold_scalar)
        )
        self.world_angular_threshold_scalar = float(
            metadata.get("settle_angular_speed_rad", self.world_angular_threshold_scalar)
        )

        self.sim_time = float(np.asarray(snapshot["sim_time"]).item())

        print(
            f"{_LOG} Loading {self.snapshot_path}: replicate world0 "
            f"({n_boxes} boxes) × {self.user_world_count} worlds, solver={self.solver_type}"
        )

        self._build_model_and_state(metadata, n_boxes)
        self._setup_removal_arrays(n_boxes)
        self._run_verify_phase()

        print(f"{_LOG} Initial verify passed.")

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
            f"{_LOG} Model ready: rigid_contact_max={rigid_contact_max}, "
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

    def _run_verify_phase(self):
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
