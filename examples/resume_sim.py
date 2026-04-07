"""
Resume Newton rigid simulation from a settled `.npz` snapshot.

Compatible with snapshots from `examples.rot_partition_sim` (array keys and `metadata_json` schema).

**Parameter priority:** body poses and counts come from the npz arrays; physics / solver /
contact budgets come from ``metadata_json`` when present. ``configs.configs_sim`` is used
only as a fallback when a metadata key is missing (or for values not stored in snapshots,
e.g. ``ENV_SPACING`` for viewer layout unless we add it to metadata later).

Rigid contact buffer: if ``rigid_contact_max`` in metadata is missing or non-positive,
``utils.world_info.estimate_rigid_contact_max`` is used (same rule as ``rot_partition_sim``).

From `unload_clean` root::

    python -m examples.resume_sim --snapshot path/to/settled.npz
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer

from configs import configs_sim as cs
from utils.create_solver import create_solver
from utils.world_info import (
    estimate_rigid_contact_max,
    load_snapshot_metadata,
    wall_definitions_for_dims,
)


class ResumeSim:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device
        self.snapshot_path = Path(args.snapshot)

        snapshot = np.load(self.snapshot_path, allow_pickle=False)
        metadata = load_snapshot_metadata(snapshot)

        self.metadata = metadata
        self.num_envs = int(metadata["world_count"])
        self.fps = int(metadata.get("fps", cs.FPS))
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(metadata.get("sim_substeps", cs.SIM_SUBSTEPS))
        self.sim_dt = float(metadata.get("sim_dt", self.frame_dt / self.sim_substeps))
        self.sim_time = float(np.asarray(snapshot["sim_time"]).item())
        self.solver_type = str(metadata.get("solver_type", cs.DEFAULT_SOLVER))

        if self.solver_type not in ("xpbd", "mujoco"):
            raise ValueError(f"Unsupported solver in snapshot metadata: {self.solver_type}")

        self.dims_np = np.asarray(metadata["dims"], dtype=np.float32)
        self.box_counts_per_world = np.asarray(snapshot["box_counts_per_world"], dtype=np.int32)
        self.body_q_np = np.asarray(snapshot["body_q"], dtype=np.float32)
        self.body_qd_np = np.asarray(snapshot["body_qd"], dtype=np.float32)
        self.body_half_extents_np = np.asarray(snapshot["body_half_extents"], dtype=np.float32)

        if self.box_counts_per_world.shape[0] != self.num_envs:
            raise ValueError(
                "Snapshot is inconsistent: box_counts_per_world length does not match world_count."
            )

        self.total_boxes = int(np.sum(self.box_counts_per_world))
        if self.total_boxes != self.body_q_np.shape[0]:
            raise ValueError(
                "Snapshot is inconsistent: sum(box_counts_per_world) does not match body_q length."
            )
        if self.body_qd_np.shape[0] != self.total_boxes or self.body_half_extents_np.shape[0] != self.total_boxes:
            raise ValueError(
                "Snapshot is inconsistent: body_qd/body_half_extents length does not match body_q."
            )

        print(f"Loading settled snapshot from {self.snapshot_path}...")
        print(
            f"Restoring {self.num_envs} worlds and {self.total_boxes} boxes "
            f"with solver={self.solver_type}."
        )

        self._build_model_from_snapshot()
        self._capture_graph()

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

    def _build_model_from_snapshot(self):
        main_builder = newton.ModelBuilder()
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(main_builder)
        except Exception as exc:
            print(f"[warn] Could not register MuJoCo custom attributes on builder: {exc}")

        rigid_gap = float(self.metadata.get("rigid_gap", cs.DEFAULT_RIGID_GAP))
        ke = float(self.metadata.get("mujoco_contact_ke", cs.MUJOCO_CONTACT_KE))
        kd_default = float(self.metadata.get("mujoco_contact_kd", cs.MUJOCO_CONTACT_KD))
        main_builder.rigid_gap = rigid_gap
        main_builder.default_shape_cfg.gap = rigid_gap
        main_builder.default_shape_cfg.ke = ke
        main_builder.default_shape_cfg.kd = kd_default

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(self.metadata.get("ground_mu", cs.GROUND_MU)),
            kd=float(self.metadata.get("ground_kd", cs.GROUND_KD)),
            mu_torsional=float(self.metadata.get("ground_mu_torsional", cs.GROUND_MU_TORSIONAL)),
            mu_rolling=float(self.metadata.get("ground_mu_rolling", cs.GROUND_MU_ROLLING)),
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
            mu=float(self.metadata.get("box_mu", cs.BOX_MU)),
            kd=float(self.metadata.get("box_kd", cs.BOX_KD)),
            mu_torsional=float(self.metadata.get("box_mu_torsional", cs.BOX_MU_TORSIONAL)),
            mu_rolling=float(self.metadata.get("box_mu_rolling", cs.BOX_MU_ROLLING)),
            gap=rigid_gap,
            ke=ke,
        )
        wall_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(self.metadata.get("wall_mu", cs.WALL_MU)),
            kd=float(self.metadata.get("wall_kd", cs.WALL_KD)),
            mu_torsional=float(self.metadata.get("wall_mu_torsional", cs.WALL_MU_TORSIONAL)),
            mu_rolling=float(self.metadata.get("wall_mu_rolling", cs.WALL_MU_ROLLING)),
            gap=rigid_gap,
            ke=ke,
        )

        dx, dy, dz = [float(v) for v in self.dims_np.tolist()]
        wall_thickness = float(self.metadata.get("wall_thickness", cs.WALL_THICKNESS))
        wall_scale = float(self.metadata.get("wall_scale", cs.WALL_SCALE))
        wr = self.metadata.get("walls_removed", [])
        walls_removed = {int(x) for x in wr} if wr else set()
        wall_defs = wall_definitions_for_dims(
            dx, dy, dz, wall_thickness, wall_scale, walls_removed=walls_removed
        )
        if walls_removed:
            print(
                f"[resume_sim] Walls omitted (indices): {sorted(walls_removed)} "
                f"=> {len(wall_defs)} wall shape(s) per env"
            )

        body_cursor = 0
        for env in range(self.num_envs):
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

            count = int(self.box_counts_per_world[env])
            for _ in range(count):
                pose = self.body_q_np[body_cursor]
                he = self.body_half_extents_np[body_cursor]
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
                body_cursor += 1

            main_builder.add_world(env_builder)

        self.model = main_builder.finalize(device=self.device)

        if self.solver_type == "mujoco":
            solimp_src = self.metadata.get("mujoco_solimp", list(cs.MUJOCO_SOLIMP))
            solimp_list = [float(x) for x in solimp_src]
            arr = self.model.mujoco.geom_solimp.numpy()
            arr[:] = np.asarray(solimp_list, dtype=arr.dtype)
            print(f"Applied MuJoCo geom_solimp={solimp_list}")

        saved_rigid_contact_max = int(self.metadata.get("rigid_contact_max", 0))
        cpb = int(self.metadata.get("contacts_per_body", cs.DEFAULT_CONTACTS_PER_BODY))
        rigid_contact_max = estimate_rigid_contact_max(
            total_bodies=self.total_boxes,
            contacts_per_body=cpb,
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

        saved_nconmax = int(self.metadata.get("nconmax", 0))
        saved_njmax = int(self.metadata.get("njmax", 0))
        nconmax = saved_nconmax
        njmax = saved_njmax
        if nconmax <= 0:
            nconmax = max(rigid_contact_max // max(self.num_envs, 1), 100)
        if njmax <= 0:
            njmax = nconmax * 3

        sk = self._solver_kwargs_from_metadata()
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
        self.state_0.body_q.assign(self.body_q_np)
        self.state_0.body_qd.assign(self.body_qd_np)
        self.state_1.assign(self.state_0)

        self.viewer.set_model(self.model)
        env_spacing = float(self.metadata.get("env_spacing", cs.ENV_SPACING))
        spacing = float(np.max(self.dims_np) + env_spacing)
        self.viewer.set_world_offsets((spacing, spacing, 0.0))

        # Top-down view (z-up): center over env grid, camera high on +Z, look toward -Z.
        max_dim = float(np.max(self.dims_np))
        cols = int(math.ceil(math.sqrt(self.num_envs)))
        rows = int(math.ceil(self.num_envs / cols))
        cx = 0.5 * float(cols - 1) * spacing
        cy = 0.5 * float(rows - 1) * spacing
        z_high = max_dim * 5.0 + spacing * 2.0
        self.viewer.set_camera(
            pos=wp.vec3(cx, cy, z_high),
            pitch=-88.0,
            yaw=0.0,
        )

        print(
            f"Snapshot restored: rigid_contact_max={rigid_contact_max}, "
            f"nconmax={nconmax}, njmax={njmax}"
        )

    def _capture_graph(self):
        if wp.get_device(self.device).is_cuda:
            saved_q = self.state_0.body_q.numpy().copy()
            saved_qd = self.state_0.body_qd.numpy().copy()
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph
            self.state_0.body_q.assign(saved_q)
            self.state_0.body_qd.assign(saved_qd)
            self.state_1.assign(self.state_0)
        else:
            self.graph = None

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if isinstance(self.viewer, newton.viewer.ViewerNull):
            self.viewer.end_frame()
        else:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()


def main():
    wp.init()
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to a .npz snapshot (e.g. from rot_partition_sim).",
    )

    viewer, args = newton.examples.init(parser)
    example = ResumeSim(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
