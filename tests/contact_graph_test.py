# Load a snapshot .npz, run the same initial stability check as unload_plan_batch (without planners/viewer),
# then run one collision query and build an undirected contact graph for a chosen world (NetworkX + matplotlib).
#
# Run from repo `unload_clean` root, e.g.:
#   python tests/contact_graph_test.py --snapshot path/to/snap.npz
#
# Requires: networkx, matplotlib (pip install networkx matplotlib)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton import GeoType

from utils.initial_verify_snapshot import InitialVerifySnapshot


def _rigid_contact_pairs_after_collide(contacts) -> tuple[int, np.ndarray, np.ndarray]:
    wp.synchronize()
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    return n, s0, s1


def _static_node_name(shape_type_np: np.ndarray | None, static_shape_idx: int) -> str:
    """Label static end of a contact: ``ground`` (PLANE/HFIELD) vs ``wall`` (e.g. BOX walls), or merged ``static``."""
    if shape_type_np is None or static_shape_idx < 0 or static_shape_idx >= len(shape_type_np):
        return "static"
    gt = int(shape_type_np[static_shape_idx])
    if gt in (int(GeoType.PLANE), int(GeoType.HFIELD)):
        return "ground"
    return "wall"


def build_world_contact_graph(
    shape_body_np: np.ndarray,
    w0: int,
    w1: int,
    s0: np.ndarray,
    s1: np.ndarray,
    shape_type_np: np.ndarray | None = None,
    static_label: str = "static",
):
    """Undirected graph: local box indices 0..n-1 plus static nodes.

    If ``shape_type_np`` is provided (``model.shape_type.numpy()``), contacts with ``body==-1``
    are split into two nodes: ``ground`` (PLANE/HFIELD) and ``wall`` (other static, e.g. BOX walls).
    Otherwise all static contacts attach to a single node ``static`` (or ``static_label``).
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise SystemExit(
            "contact_graph_test requires networkx. Install with: pip install networkx matplotlib"
        ) from e

    G = nx.Graph()
    n_dyn = w1 - w0
    for i in range(n_dyn):
        G.add_node(i, kind="box")

    def in_world(b: int) -> bool:
        return b >= 0 and w0 <= b < w1

    seen = set()
    for a, b in zip(s0, s1):
        ia, ib = int(a), int(b)
        ba = int(shape_body_np[ia])
        bb = int(shape_body_np[ib])
        if ba == bb:
            continue
        if not (in_world(ba) or in_world(bb)):
            continue
        if ba == -1 and bb == -1:
            continue
        if in_world(ba) and in_world(bb):
            u, v = ba - w0, bb - w0
            if u > v:
                u, v = v, u
            key = ("bb", u, v)
        elif in_world(ba) and bb == -1:
            sk = _static_node_name(shape_type_np, ib)
            key = ("bs", ba - w0, sk)
        elif in_world(bb) and ba == -1:
            sk = _static_node_name(shape_type_np, ia)
            key = ("bs", bb - w0, sk)
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        if key[0] == "bb":
            _, u, v = key
            G.add_edge(u, v, kind="box-box")
        else:
            _, u, sk = key
            if shape_type_np is None:
                node = static_label
                kind = "static"
            else:
                node = sk
                kind = sk
            G.add_node(node, kind=kind)
            G.add_edge(u, node, kind=f"box-{kind}")

    return G


def draw_graph(G, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise SystemExit(
            "Drawing requires matplotlib. Install with: pip install matplotlib networkx"
        ) from e

    pos = nx.spring_layout(G, seed=0, k=0.9, iterations=50)
    plt.figure(figsize=(10, 8))
    color_map = {
        "box": "#6fa8dc",
        "ground": "#7dcea0",
        "wall": "#d4ac6e",
        "static": "#aeb6bf",
    }
    node_colors = [color_map.get(G.nodes[n].get("kind", "box"), "#6fa8dc") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, alpha=0.7)
    plt.title(title)
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[contact_graph_test] Wrote {out_path}")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--snapshot", type=str, required=True, help="Path to .npz snapshot")
    parser.add_argument("--world-count", "--ne", type=int, default=1, help="Replicated worlds (default: 1)")
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=10,
        help="Full frames before stability check (same as unload_plan_batch --verify-steps)",
    )
    parser.add_argument(
        "--verify-threshold-scale",
        type=float,
        default=1.0,
        help="Threshold scale for verify only (same as unload_plan_batch)",
    )
    parser.add_argument(
        "--settle-cooldown-frames",
        type=int,
        default=-1,
        help="Passed through for argparse compatibility; unused by this script.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="contact_graph_world0.png",
        help="PNG path for NetworkX layout figure",
    )
    parser.add_argument(
        "--world-index",
        type=int,
        default=0,
        help="Which world to extract (default: 0). Uses body_world_start[world]..[world+1].",
    )

    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)
    wp.init()

    print("[contact_graph_test] Loading snapshot, building scene, initial verify (unload_plan_batch semantics)...")
    sim = InitialVerifySnapshot(args)

    widx = int(args.world_index)
    if widx < 0 or widx >= sim.world_count:
        print(f"[contact_graph_test] world-index {widx} out of range [0, {sim.world_count})", file=sys.stderr)
        sys.exit(1)

    bws = sim.body_world_start_np
    w0 = int(bws[widx])
    w1 = int(bws[widx + 1])

    print("[contact_graph_test] Running collision pipeline for current state (post-verify)...")
    sim.collision_pipeline.collide(sim.state_0, sim.contacts)
    n_tot, s0, s1 = _rigid_contact_pairs_after_collide(sim.contacts)

    shape_body_np = sim.shape_body_np
    st = getattr(sim.model, "shape_type", None)
    shape_type_np = st.numpy() if st is not None else None
    G = build_world_contact_graph(shape_body_np, w0, w1, s0, s1, shape_type_np=shape_type_np)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(
        f"[contact_graph_test] rigid contacts (all shapes): {n_tot} | "
        f"world {widx} graph: nodes={n_nodes}, edges={n_edges}"
    )

    out = Path(args.output)
    title = f"Contact graph — world {widx} ({n_edges} edges, {n_tot} raw contacts)"
    draw_graph(G, out, title)


if __name__ == "__main__":
    main()
