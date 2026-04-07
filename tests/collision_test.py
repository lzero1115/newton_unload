# One world, two stacked boxes: after initial settle, record bottom pose as graveyard;
# after +3 s sim time, set bottom shape collision group to 0 and substep-enforce that pose;
# run until the top box is stable.

import math

import numpy as np
import warp as wp

import newton
import newton.examples

FPS = 60
SIM_SUBSTEPS = 10
XPBD_ITERATIONS = 8
XPBD_CONTACT_RELAXATION = 0.9
ANGULAR_DAMPING = 0.1

BOTTOM_HALF = (0.08, 0.08, 0.08)
TOP_HALF = (0.06, 0.06, 0.06)
Z0 = 0.01

SETTLE_LIN = 0.02
SETTLE_ANG = 0.02
SETTLE_FRAMES_REQUIRED = 30
TOP_SETTLE_FRAMES_REQUIRED = 45

DELAY_BEFORE_GHOST_S = 3.0

MIN_RIGID_CONTACT_MAX = 5000
CONTACTS_PER_BODY_EST = 40


class CollisionGroupTestSim:
    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.device = args.device

        self.settle_lin_sq = SETTLE_LIN ** 2
        self.settle_ang_sq = SETTLE_ANG ** 2

        main = newton.ModelBuilder()
        main.rigid_gap = 0.005
        main.default_shape_cfg.gap = 0.005
        main.add_shape_plane(
            body=-1,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
        )

        hx0, hy0, hz0 = BOTTOM_HALF
        hx1, hy1, hz1 = TOP_HALF

        wb = newton.ModelBuilder()
        wb.rigid_gap = 0.005
        wb.default_shape_cfg.gap = 0.005

        z = Z0
        z += hz0
        b0 = wb.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()))
        wb.add_shape_box(b0, hx=float(hx0), hy=float(hy0), hz=float(hz0))
        z += hz0

        z += hz1
        b1 = wb.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()))
        wb.add_shape_box(b1, hx=float(hx1), hy=float(hy1), hz=float(hz1))
        z += hz1

        main.add_world(wb)
        self.model = main.finalize(device=self.device)

        self.body_world_start_np = self.model.body_world_start.numpy()
        w0 = int(self.body_world_start_np[0])
        w1 = int(self.body_world_start_np[1])
        self.bottom_body = w0
        self.top_body = w0 + 1
        assert self.top_body < w1

        self.shape_body_np = self.model.shape_body.numpy()
        self.body_shape_indices = self._shapes_for_body(self.bottom_body)

        total_bodies = self.model.body_count
        rigid_contact_max = max(
            total_bodies * CONTACTS_PER_BODY_EST,
            MIN_RIGID_CONTACT_MAX,
        )
        self.model.rigid_contact_max = rigid_contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=rigid_contact_max,
        )
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=XPBD_ITERATIONS,
            rigid_contact_relaxation=XPBD_CONTACT_RELAXATION,
            angular_damping=ANGULAR_DAMPING,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.graveyard_body_q_row = None

        self.phase = "initial_settling"
        self.sim_time = 0.0
        self.time_at_initial_settle = None
        self.time_bottom_collision_disabled = None
        self.time_top_stable = None
        self.frames_stable_initial = 0
        self.frames_stable_top = 0

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.5, 1.5, 0.0))
        self.viewer.set_camera(
            pos=wp.vec3(1.2, -1.2, 0.9),
            pitch=-18.0,
            yaw=135.0,
        )

        print(
            f"[collision_test] bottom_body={self.bottom_body} top_body={self.top_body}, "
            f"DELAY_BEFORE_GHOST_S={DELAY_BEFORE_GHOST_S}"
        )

    def _shapes_for_body(self, body_index: int):
        out = []
        for si, bi in enumerate(self.shape_body_np):
            if int(bi) == body_index:
                out.append(si)
        return out

    def _body_speed_sq(self, body_qd_np: np.ndarray, body_idx: int) -> tuple[float, float]:
        row = body_qd_np[body_idx]
        lv = row[0:3]
        av = row[3:6]
        return float(np.dot(lv, lv)), float(np.dot(av, av))

    def _both_initially_stable(self, body_qd_np: np.ndarray) -> bool:
        for bid in (self.bottom_body, self.top_body):
            ls, asq = self._body_speed_sq(body_qd_np, bid)
            if ls >= self.settle_lin_sq or asq >= self.settle_ang_sq:
                return False
        return True

    def _top_stable(self, body_qd_np: np.ndarray) -> bool:
        ls, asq = self._body_speed_sq(body_qd_np, self.top_body)
        return ls < self.settle_lin_sq and asq < self.settle_ang_sq

    def _disable_bottom_collision(self):
        col = self.model.shape_collision_group.numpy().copy()
        for si in self.body_shape_indices:
            col[si] = 0
        self.model.shape_collision_group.assign(col)
        shapes_str = ", ".join(str(s) for s in self.body_shape_indices)
        dt_from_settle = (
            float(self.sim_time) - float(self.time_at_initial_settle)
            if self.time_at_initial_settle is not None
            else float("nan")
        )
        print(
            "[collision_test] Bottom box: collision group set to 0 (no collision). "
            f"body_index={self.bottom_body}, shape_indices=[{shapes_str}], "
            f"sim_t={self.sim_time:.4f}s "
            f"(elapsed since initial settle {dt_from_settle:.4f}s / target delay {DELAY_BEFORE_GHOST_S}s)"
        )

    def _enforce_bottom_graveyard(self):
        if self.graveyard_body_q_row is None:
            return
        bq = self.state_0.body_q.numpy().copy()
        bqd = self.state_0.body_qd.numpy().copy()
        bq[self.bottom_body] = self.graveyard_body_q_row
        bqd[self.bottom_body] = 0.0
        self.state_0.body_q.assign(bq)
        self.state_0.body_qd.assign(bqd)

    def _simulate_substeps(self):
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
            if self.phase == "ghost_bottom":
                self._enforce_bottom_graveyard()

    def step(self):
        if self.phase == "done":
            return

        self._simulate_substeps()
        self.sim_time += self.frame_dt

        body_qd_np = self.state_0.body_qd.numpy()

        if self.phase == "initial_settling":
            if self._both_initially_stable(body_qd_np):
                self.frames_stable_initial += 1
            else:
                self.frames_stable_initial = 0
            if self.frames_stable_initial >= SETTLE_FRAMES_REQUIRED:
                g = self.state_0.body_q.numpy().copy()
                self.graveyard_body_q_row = g[self.bottom_body].copy()
                self.time_at_initial_settle = float(self.sim_time)
                self.phase = "wait_3s"
                print(
                    "[collision_test] Initial stack settled: "
                    f"sim_t={self.sim_time:.4f}s, graveyard pose stored for bottom body={self.bottom_body}; "
                    f"will set bottom collision group to 0 after {DELAY_BEFORE_GHOST_S}s more sim time."
                )
            return

        if self.phase == "wait_3s":
            if self.sim_time >= self.time_at_initial_settle + DELAY_BEFORE_GHOST_S:
                self._disable_bottom_collision()
                self.time_bottom_collision_disabled = float(self.sim_time)
                self.phase = "ghost_bottom"
                self.frames_stable_top = 0
                print(
                    "[collision_test] Ghost phase: bottom body pinned to graveyard each substep; "
                    f"watching top body={self.top_body} until stable ("
                    f"{TOP_SETTLE_FRAMES_REQUIRED} consecutive frames below threshold)."
                )
            return

        if self.phase == "ghost_bottom":
            if self._top_stable(body_qd_np):
                self.frames_stable_top += 1
            else:
                self.frames_stable_top = 0
            if self.frames_stable_top >= TOP_SETTLE_FRAMES_REQUIRED:
                self.phase = "done"
                self.time_top_stable = float(self.sim_time)
                t1 = self.time_bottom_collision_disabled
                if t1 is not None:
                    dt_ghost_to_top = self.time_top_stable - t1
                else:
                    dt_ghost_to_top = float("nan")
                t0 = self.time_at_initial_settle
                print(
                    "[collision_test] Remaining box (top) stable: "
                    f"sim_t={self.time_top_stable:.4f}s "
                    f"(body_index={self.top_body}, {TOP_SETTLE_FRAMES_REQUIRED} consecutive frames below threshold)."
                )
                print(
                    "[collision_test] Timeline: "
                    f"t0_initial_settle={t0:.4f}s; "
                    f"t1_bottom_collision_group_0={t1:.4f}s; "
                    f"t2_top_stable={self.time_top_stable:.4f}s; "
                    f"delta(t1 -> t2)={dt_ghost_to_top:.4f}s."
                )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main():
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = CollisionGroupTestSim(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
