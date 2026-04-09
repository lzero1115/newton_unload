"""Launch Warp kernels to fill world-space box corner buffers (dynamic bodies + static walls)."""

from __future__ import annotations

import numpy as np
import warp as wp

from kernels.mesh_extract_kernels import (
    extract_box_vertices_world,
    extract_static_box_vertices_world,
)


class PartitionSceneMeshGpu:
    """Parallel vertex extraction: dynamic boxes and static walls both use 2D + prefix arrays (``sim_kernels`` style)."""

    def __init__(
        self,
        device,
        body_world_start_np: np.ndarray,
        max_bodies_per_world: int,
        half_extents_batched_np: np.ndarray,
        wall_defs: list,
    ):
        self.device = device
        bws = np.asarray(body_world_start_np, dtype=np.int32).reshape(-1)
        self.num_worlds = int(len(bws) - 1)
        self.max_bodies_per_world = int(max_bodies_per_world)
        self.num_walls = len(wall_defs)
        self.num_wall_segments = 1 if self.num_walls > 0 else 0
        self.max_walls_per_segment = int(self.num_walls) if self.num_walls > 0 else 0

        he_b = np.asarray(half_extents_batched_np, dtype=np.float32)
        if he_b.ndim != 2 or he_b.shape[1] != 3:
            raise ValueError(f"half_extents_batched_np expected (N, 3), got {he_b.shape}")
        expected_n = self.num_worlds * self.max_bodies_per_world
        if int(he_b.shape[0]) != expected_n:
            raise ValueError(
                f"half_extents_batched_np expected ({expected_n}, 3) = worlds*max_bodies, got {he_b.shape}"
            )

        self._body_world_start = wp.array(bws, dtype=wp.int32, device=device)
        self._half_batched = wp.array(he_b, dtype=wp.vec3, device=device)
        self._out_boxes = wp.zeros(
            self.num_worlds * self.max_bodies_per_world * 8, dtype=wp.vec3, device=device
        )

        if self.num_walls > 0:
            xf_list = []
            wh_list = []
            for pos, whx, why, whz in wall_defs:
                p0, p1, p2 = float(pos[0]), float(pos[1]), float(pos[2])
                xf_list.append(wp.transform(wp.vec3(p0, p1, p2), wp.quat_identity()))
                wh_list.append(wp.vec3(float(whx), float(why), float(whz)))
            self._wall_xf = wp.array(xf_list, dtype=wp.transform, device=device)
            self._wall_half = wp.array(wh_list, dtype=wp.vec3, device=device)
            seg_start = np.array([0, self.num_walls], dtype=np.int32)
            self._static_segment_start = wp.array(seg_start, dtype=wp.int32, device=device)
            self._out_walls = wp.zeros(
                self.num_wall_segments * self.max_walls_per_segment * 8, dtype=wp.vec3, device=device
            )
        else:
            self._wall_xf = None
            self._wall_half = None
            self._static_segment_start = None
            self._out_walls = None

    def launch_dynamic_boxes(self, body_q: wp.array) -> None:
        wp.launch(
            extract_box_vertices_world,
            dim=(self.num_worlds, self.max_bodies_per_world),
            inputs=[
                body_q,
                self._half_batched,
                self._body_world_start,
                int(self.max_bodies_per_world),
                self._out_boxes,
            ],
            device=self.device,
        )

    def box_vertex_slice(self, world_id: int, body_local_id: int) -> slice:
        """Slice into ``box_vertices_numpy()`` for one dynamic box (8 verts)."""
        base = (world_id * self.max_bodies_per_world + body_local_id) * 8
        return slice(base, base + 8)

    def wall_vertex_slice(self, segment_id: int, wall_local_id: int) -> slice:
        """Slice into ``wall_vertices_numpy()`` for one wall box (8 verts)."""
        base = (segment_id * self.max_walls_per_segment + wall_local_id) * 8
        return slice(base, base + 8)

    def launch_static_walls(self) -> None:
        if self.num_walls <= 0 or self._out_walls is None or self._static_segment_start is None:
            return
        wp.launch(
            extract_static_box_vertices_world,
            dim=(self.num_wall_segments, self.max_walls_per_segment),
            inputs=[
                self._wall_xf,
                self._wall_half,
                self._static_segment_start,
                int(self.max_walls_per_segment),
                self._out_walls,
            ],
            device=self.device,
        )

    def box_vertices_numpy(self) -> np.ndarray:
        wp.synchronize()
        v = self._out_boxes.numpy()
        return np.asarray(v, dtype=np.float64)

    def wall_vertices_numpy(self) -> np.ndarray | None:
        if self._out_walls is None:
            return None
        wp.synchronize()
        v = self._out_walls.numpy()
        return np.asarray(v, dtype=np.float64)
