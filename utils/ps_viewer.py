# Polyscope viewer for PsPartitionSim (``examples.ps_partition_sim``; lazy-import polyscope in run function).
# Half extents: ``sim.viz_half_extents_batched_np`` (partition output).

from __future__ import annotations

import numpy as np
import warp as wp

from configs import configs_sim as cs
from utils.gpu_partition_mesh_extract import PartitionSceneMeshGpu
from utils.world_info import wall_definitions_for_dims

_CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [6, 5, 4],
        [7, 6, 4],
        [1, 5, 6],
        [1, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [3, 2, 6],
        [3, 6, 7],
        [1, 0, 4],
        [1, 4, 5],
    ],
    dtype=np.int32,
)

_COLOR_BOX_INSTANT_SETTLED = (0.58, 0.58, 0.6)
_COLOR_BOX_INSTANT_UNSETTLED = (0.92, 0.1, 0.1)


def run_polyscope_rot_partition(sim) -> None:
    """``sim`` must be a ``PsPartitionSim`` instance (``examples.ps_partition_sim``)."""
    import polyscope as ps
    import polyscope.imgui as psim

    dims = sim.dims_np
    wall_defs = wall_definitions_for_dims(
        float(dims[0]),
        float(dims[1]),
        float(dims[2]),
        float(cs.WALL_THICKNESS),
        float(cs.WALL_SCALE),
        walls_removed=set(sim.walls_removed),
    )

    extractor = PartitionSceneMeshGpu(
        sim.model.device,
        sim.body_world_start_np,
        sim.max_bodies_per_world,
        sim.viz_half_extents_batched_np,
        wall_defs,
    )

    box_mesh_handles: list = []

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_program_name("ps_partition_sim")

    extractor.launch_static_walls()
    wall_np_full = extractor.wall_vertices_numpy()

    if wall_np_full is not None and extractor.num_walls > 0:
        for wi in range(extractor.num_walls):
            verts = wall_np_full[extractor.wall_vertex_slice(0, wi)]
            m = ps.register_surface_mesh(f"wall_{wi}", verts, _CUBE_FACES)
            m.set_color((0.55, 0.36, 0.22))
            m.set_transparency(0.5)

    extractor.launch_dynamic_boxes(sim.state_0.body_q)
    wp.synchronize()
    full0 = extractor.box_vertices_numpy()
    num_w = int(len(sim.box_counts_per_world))
    for w in range(num_w):
        n = int(sim.box_counts_per_world[w])
        for i in range(n):
            verts8 = full0[extractor.box_vertex_slice(w, i)].copy()
            m = ps.register_surface_mesh(f"box_w{w}_{i}", verts8, _CUBE_FACES)
            m.set_color(_COLOR_BOX_INSTANT_SETTLED)
            box_mesh_handles.append(m)

    def callback() -> None:
        psim.TextUnformatted(
            "Scene: settled (simulation stopped)"
            if sim.all_settled
            else "Scene: running"
        )
        psim.Separator()

        if sim.all_settled:
            for mh in box_mesh_handles:
                mh.set_color(_COLOR_BOX_INSTANT_SETTLED)
            if sim.snapshot_path and not sim.snapshot_saved:
                sim._save_snapshot()
            return
        sim.step()
        sim.refresh_instant_body_unsettled_flags()
        wp.synchronize()
        body_u = sim.body_instant_unsettled.numpy()
        extractor.launch_dynamic_boxes(sim.state_0.body_q)
        wp.synchronize()
        full = extractor.box_vertices_numpy()
        bws = sim.body_world_start_np
        hi = 0
        for w in range(num_w):
            w0 = int(bws[w])
            n = int(sim.box_counts_per_world[w])
            for i in range(n):
                gid = w0 + i
                box_mesh_handles[hi].set_color(
                    _COLOR_BOX_INSTANT_UNSETTLED
                    if int(body_u[gid]) != 0
                    else _COLOR_BOX_INSTANT_SETTLED
                )
                box_mesh_handles[hi].update_vertex_positions(full[extractor.box_vertex_slice(w, i)])
                hi += 1

    ps.set_user_callback(callback)
    ps.show()
