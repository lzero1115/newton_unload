"""Construct Newton XPBD or MuJoCo solver from explicit hyperparameters (no config coupling)."""

from __future__ import annotations

import newton


def create_solver(
    model,
    solver_type: str,
    nconmax: int = 500,
    njmax: int = 1500,
    *,
    xpbd_iterations: int,
    xpbd_contact_relaxation: float,
    xpbd_angular_damping: float,
    xpbd_enable_restitution: bool,
    mujoco_iterations: int,
    mujoco_ls_iterations: int,
    mujoco_solver: str,
    mujoco_integrator: str,
    mujoco_cone: str,
    mujoco_impratio: float,
    mujoco_tolerance: float,
    mujoco_ls_tolerance: float,
    mujoco_update_data_interval: int,
    mujoco_use_contacts: bool,
):
    if solver_type == "xpbd":
        return newton.solvers.SolverXPBD(
            model,
            iterations=xpbd_iterations,
            rigid_contact_relaxation=xpbd_contact_relaxation,
            angular_damping=xpbd_angular_damping,
            enable_restitution=xpbd_enable_restitution,
        )
    if solver_type == "mujoco":
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=mujoco_use_contacts,
            nconmax=nconmax,
            njmax=njmax,
            iterations=mujoco_iterations,
            ls_iterations=mujoco_ls_iterations,
            solver=mujoco_solver,
            integrator=mujoco_integrator,
            cone=mujoco_cone,
            impratio=mujoco_impratio,
            tolerance=mujoco_tolerance,
            ls_tolerance=mujoco_ls_tolerance,
            update_data_interval=mujoco_update_data_interval,
        )
    raise ValueError(f"Unknown solver: {solver_type}. Choose 'xpbd' or 'mujoco'.")
