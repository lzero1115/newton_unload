"""
Multi-world partition rigid sim: `--solver` on the CLI; other settings from `configs.configs_ps`.

Partition driver: `examples.partition_batch_example.batched_partition`.

From `unload_clean` root::

    python -m examples.partition_sim_example

Extra CLI flag: `--solver` only (rest from `newton.examples.create_parser()`).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

from configs import configs_ps
from examples.partition_batch_example import batched_partition
from utils import create_solver
from utils.world_info import estimate_rigid_contact_max


def validate_partition_results(results, num_envs: int, expected_boxes: int, solver_type: str):
    if len(results) != num_envs:
        raise ValueError(f"Expected {num_envs} environments, but partition returned {len(results)}.")

    counts = [len(env_boxes) for env_boxes in results]
    if not counts:
        raise ValueError("Partition returned no environments.")

    unique_counts = sorted(set(counts))
    if solver_type != "mujoco":
        return

    if len(unique_counts) != 1:
        preview = ", ".join(f"env {i}: {count}" for i, count in enumerate(counts[:8]))
        raise ValueError(
            "MuJoCo requires homogeneous worlds, but the partition produced different box counts per world. "
            f"Requested {expected_boxes} boxes per world; got counts {unique_counts}. "
            f"First counts: {preview}. Adjust MIN_RATIO / DIMS / NB in configs_ps.py."
        )

    actual_boxes = unique_counts[0]
    if actual_boxes != expected_boxes:
        raise ValueError(
            "MuJoCo homogeneity check passed, but the partition did not reach the requested box count. "
            f"Requested {expected_boxes} boxes per world; got {actual_boxes}. "
            "Adjust MIN_RATIO / DIMS / NB in configs_ps.py."
        )


class PartitionSim:
    def __init__(self, viewer, device: str, solver: str):
        self.fps = configs_ps.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = configs_ps.SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.solver_type = solver
        self.drop_height = configs_ps.DROP_HEIGHT

        num_envs = configs_ps.NUM_ENVS
        dims = list(configs_ps.DIMS)
        n_target = configs_ps.NB
        min_ratio = configs_ps.MIN_RATIO
        shrink = configs_ps.SHRINK
        seed = configs_ps.SEED

        print(f"Running batched partition: {num_envs} envs, {n_target} boxes each...")
        results, dims_np = batched_partition(
            num_envs=num_envs,
            dims=dims,
            n_target=n_target,
            min_ratio=min_ratio,
            shrink_factor=shrink,
            seed=seed,
            device=device,
        )
        validate_partition_results(results, num_envs, n_target, self.solver_type)
        print("Partition done. Building Newton model...")

        main_builder = newton.ModelBuilder()
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(main_builder)
        except Exception as exc:
            print(f"[warn] Could not register MuJoCo custom attributes on builder: {exc}")

        main_builder.rigid_gap = configs_ps.DEFAULT_RIGID_GAP
        main_builder.default_shape_cfg.gap = configs_ps.DEFAULT_RIGID_GAP
        main_builder.default_shape_cfg.ke = configs_ps.MUJOCO_CONTACT_KE
        main_builder.default_shape_cfg.kd = configs_ps.MUJOCO_CONTACT_KD
        main_builder.default_shape_cfg.restitution = configs_ps.SHAPE_RESTITUTION

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=configs_ps.GROUND_MU,
            gap=configs_ps.DEFAULT_RIGID_GAP,
            ke=configs_ps.MUJOCO_CONTACT_KE,
            kd=configs_ps.MUJOCO_CONTACT_KD,
            restitution=configs_ps.SHAPE_RESTITUTION,
        )
        main_builder.add_shape_plane(
            body=-1,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=ground_cfg,
        )

        max_dim = np.max(dims_np)

        for env_boxes in results:
            env_builder = newton.ModelBuilder()
            env_builder.rigid_gap = configs_ps.DEFAULT_RIGID_GAP
            env_builder.default_shape_cfg.gap = configs_ps.DEFAULT_RIGID_GAP
            env_builder.default_shape_cfg.ke = configs_ps.MUJOCO_CONTACT_KE
            env_builder.default_shape_cfg.kd = configs_ps.MUJOCO_CONTACT_KD
            env_builder.default_shape_cfg.restitution = configs_ps.SHAPE_RESTITUTION

            for center, lengths in env_boxes:
                cx, cy, cz = center
                lx, ly, lz = lengths
                spawn_z = cz + self.drop_height
                body = env_builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(cx), float(cy), float(spawn_z)),
                        q=wp.quat_identity(),
                    )
                )
                env_builder.add_shape_box(
                    body,
                    hx=float(lx / 2),
                    hy=float(ly / 2),
                    hz=float(lz / 2),
                )
            main_builder.add_world(env_builder)

        self.model = main_builder.finalize(device=device)

        if self.solver_type == "mujoco":
            solimp_list = list(configs_ps.MUJOCO_SOLIMP)
            arr = self.model.mujoco.geom_solimp.numpy()
            arr[:] = np.asarray(solimp_list, dtype=arr.dtype)
            print(f"Applied MuJoCo geom_solimp={solimp_list}")

        total_bodies = sum(len(env_boxes) for env_boxes in results)
        rigid_contact_max = estimate_rigid_contact_max(
            total_bodies=total_bodies,
            contacts_per_body=configs_ps.DEFAULT_CONTACTS_PER_BODY,
            user_override=configs_ps.RIGID_CONTACT_MAX,
            min_rigid_contact_max=configs_ps.MIN_RIGID_CONTACT_MAX,
        )
        self.model.rigid_contact_max = rigid_contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=rigid_contact_max,
        )

        nconmax = configs_ps.NCONMAX if configs_ps.NCONMAX > 0 else max(rigid_contact_max // max(num_envs, 1), 100)
        njmax = configs_ps.NJMAX if configs_ps.NJMAX > 0 else nconmax * 3
        print(
            f"Using contact budgets: rigid_contact_max={rigid_contact_max}, "
            f"nconmax={nconmax}, njmax={njmax}"
        )

        self.solver = create_solver(
            self.model,
            self.solver_type,
            nconmax=nconmax,
            njmax=njmax,
            xpbd_iterations=configs_ps.XPBD_ITERATIONS,
            xpbd_contact_relaxation=configs_ps.XPBD_CONTACT_RELAXATION,
            xpbd_angular_damping=configs_ps.XPBD_ANGULAR_DAMPING,
            xpbd_enable_restitution=configs_ps.XPBD_ENABLE_RESTITUTION,
            mujoco_iterations=configs_ps.MUJOCO_ITERATIONS,
            mujoco_ls_iterations=configs_ps.MUJOCO_LS_ITERATIONS,
            mujoco_solver=configs_ps.MUJOCO_SOLVER,
            mujoco_integrator=configs_ps.MUJOCO_INTEGRATOR,
            mujoco_cone=configs_ps.MUJOCO_CONE,
            mujoco_impratio=configs_ps.MUJOCO_IMPRATIO,
            mujoco_tolerance=configs_ps.MUJOCO_TOLERANCE,
            mujoco_ls_tolerance=configs_ps.MUJOCO_LS_TOLERANCE,
            mujoco_update_data_interval=configs_ps.MUJOCO_UPDATE_DATA_INTERVAL,
            mujoco_use_contacts=configs_ps.MUJOCO_USE_CONTACTS,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        spacing = float(max_dim + configs_ps.ENV_SPACING)
        self.viewer.set_world_offsets((spacing, spacing, 0.0))

        cols = int(math.ceil(math.sqrt(num_envs)))
        cam_dist = cols * spacing * 0.9
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.6),
            pitch=-20.0,
            yaw=135.0,
        )

        self.capture()
        print(f"Newton model ready (solver={self.solver_type}). Starting simulation...")

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default=configs_ps.DEFAULT_SOLVER,
        choices=["xpbd", "mujoco"],
        help="Rigid solver; iteration/contact/buffer settings from configs_ps.",
    )
    viewer, args = newton.examples.init(parser)
    example = PartitionSim(viewer, args.device, args.solver)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
