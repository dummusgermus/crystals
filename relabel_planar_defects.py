"""Relabel planar graphs with an alternate stack→defect-atom mapping.

Copies geometry, targets, edges, and cycle features from an existing
``planar_pyg_dataset.pt`` and recomputes only:

* node ``is_defect`` / ``dist_to_defect``
* edge ``incident_defect``

This keeps the original baseline dataset intact while producing new datasets
for alternate label definitions (C14/C15 deviation, matrix-aligned, …).

Particle ids in the Laves planar archive are sequential ``1..N`` matching node
order, so id→index is ``id - 1``.  Distances use minimum-image PBC when a
matching ``basefile.data`` is found under ``--simulations-dir``.

Example::

    python relabel_planar_defects.py \\
        --defect-atoms-json laves_defect_atoms_c14c15.json \\
        --output planar_pyg_dataset_c14c15.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def load_defect_atoms_by_stack(json_path: str) -> Dict[str, List[int]]:
    """Load ``{stack_sequence: [atom_id, ...]}`` from a defect-atoms JSON."""
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    mapping = payload.get("defect_atoms", payload)
    return {
        str(stack): [int(a) for a in ids]
        for stack, ids in mapping.items()
        if not str(stack).startswith("_")
    }


BOX_RE = re.compile(
    r"^\s*([^\s]+)\s+([^\s]+)\s+xlo\s+xhi\s*$", re.IGNORECASE
)
YBOX_RE = re.compile(
    r"^\s*([^\s]+)\s+([^\s]+)\s+ylo\s+yhi\s*$", re.IGNORECASE
)
ZBOX_RE = re.compile(
    r"^\s*([^\s]+)\s+([^\s]+)\s+zlo\s+zhi\s*$", re.IGNORECASE
)
TILT_RE = re.compile(
    r"^\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+xy\s+xz\s+yz\s*$", re.IGNORECASE
)


def _parse_lammps_cell(basefile_path: str) -> Optional[Tuple[np.ndarray, Tuple[bool, bool, bool]]]:
    """Return (cell_matrix, pbc) from a LAMMPS data file, or None on failure."""
    try:
        with open(basefile_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    for line in lines:
        m = BOX_RE.match(line)
        if m:
            xlo, xhi = float(m.group(1)), float(m.group(2))
            continue
        m = YBOX_RE.match(line)
        if m:
            ylo, yhi = float(m.group(1)), float(m.group(2))
            continue
        m = ZBOX_RE.match(line)
        if m:
            zlo, zhi = float(m.group(1)), float(m.group(2))
            continue
        m = TILT_RE.match(line)
        if m:
            xy, xz, yz = float(m.group(1)), float(m.group(2)), float(m.group(3))

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        return None

    lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )
    return cell, (True, True, True)


def _min_image_distances(
    positions: np.ndarray,
    defect_indices: List[int],
    cell: Optional[np.ndarray],
) -> np.ndarray:
    """Per-node distance to the nearest defect atom (min-image if *cell* given)."""
    n = positions.shape[0]
    dist = np.zeros(n, dtype=float)
    if not defect_indices:
        return dist

    defect_pos = positions[defect_indices]
    if cell is None:
        for i in range(n):
            deltas = defect_pos - positions[i]
            dist[i] = float(np.linalg.norm(deltas, axis=1).min())
        return dist

    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        inv_cell = np.linalg.pinv(cell)

    frac = positions @ inv_cell
    defect_frac = frac[defect_indices]
    for i in range(n):
        dfrac = defect_frac - frac[i]
        dfrac -= np.round(dfrac)
        deltas = dfrac @ cell
        dist[i] = float(np.linalg.norm(deltas, axis=1).min())
    return dist


def _ids_to_indices(defect_ids: List[int], num_nodes: int, folder: str) -> List[int]:
    """Map 1-based LAMMPS ids to node indices (sequential archive layout)."""
    indices: List[int] = []
    for did in defect_ids:
        idx = int(did) - 1
        if idx < 0 or idx >= num_nodes:
            raise ValueError(
                f"{folder}: defect id {did} out of range for {num_nodes} atoms"
            )
        indices.append(idx)
    return indices


def relabel_graph(
    graph: Data,
    defect_ids: Optional[List[int]],
    cell: Optional[np.ndarray] = None,
) -> Data:
    """Return a cloned graph with updated defect node/edge features."""
    new_graph = graph.clone()
    n = int(new_graph.num_nodes)
    x = new_graph.x.clone()
    edge_index = new_graph.edge_index
    edge_attr = new_graph.edge_attr.clone()

    if defect_ids is None:
        defect_ids = []

    defect_indices = _ids_to_indices(defect_ids, n, getattr(graph, "folder", "?"))
    defect_set = set(defect_indices)
    positions = new_graph.pos.detach().cpu().numpy()
    dist = _min_image_distances(positions, defect_indices, cell)

    is_defect = torch.zeros(n, dtype=torch.float)
    for idx in defect_indices:
        is_defect[idx] = 1.0
    x[:, 2] = is_defect
    x[:, 3] = torch.tensor(dist, dtype=torch.float)

    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        incident = torch.tensor(
            [1.0 if (i in defect_set or j in defect_set) else 0.0 for i, j in zip(src, dst)],
            dtype=torch.float,
        )
        edge_attr[:, 2] = incident

    new_graph.x = x
    new_graph.edge_attr = edge_attr

    meta = dict(getattr(graph, "meta", {}) or {})
    meta["has_defect_labels"] = bool(defect_indices)
    meta["num_defect_atoms"] = len(defect_indices)
    meta["defect_ids"] = list(defect_ids)
    new_graph.meta = meta
    new_graph.defect_ids = list(defect_ids)
    return new_graph


def relabel_dataset(
    dataset: List[Data],
    defect_atoms_by_stack: Dict[str, List[int]],
    simulations_dir: Optional[str] = None,
) -> Tuple[List[Data], Dict]:
    """Relabel every graph; return new list and summary stats."""
    cell_cache: Dict[str, Optional[np.ndarray]] = {}
    augmented: List[Data] = []
    unknown_stacks = set()
    per_stack_defect_atoms: Dict[str, int] = {}

    for graph in dataset:
        stack = getattr(graph, "stack_sequence", None)
        folder = getattr(graph, "folder", None)
        if stack not in defect_atoms_by_stack:
            unknown_stacks.add(stack)
            defect_ids: Optional[List[int]] = None
        else:
            defect_ids = list(defect_atoms_by_stack[stack])

        cell = None
        if simulations_dir and folder:
            if folder not in cell_cache:
                basefile = os.path.join(simulations_dir, folder, "basefile.data")
                parsed = _parse_lammps_cell(basefile) if os.path.isfile(basefile) else None
                cell_cache[folder] = parsed[0] if parsed else None
            cell = cell_cache[folder]

        new_graph = relabel_graph(graph, defect_ids, cell=cell)
        augmented.append(new_graph)
        if stack is not None:
            per_stack_defect_atoms[stack] = int(new_graph.meta["num_defect_atoms"])

    stats = {
        "num_graphs": len(augmented),
        "graphs_with_defect_labels": sum(
            1 for g in augmented if g.meta.get("has_defect_labels", False)
        ),
        "total_defect_atoms": sum(
            g.meta.get("num_defect_atoms", 0) for g in augmented
        ),
        "defect_atoms_per_stack": per_stack_defect_atoms,
        "unknown_stacks": sorted(s for s in unknown_stacks if s is not None),
        "mapping_stacks": sorted(defect_atoms_by_stack.keys()),
    }
    return augmented, stats


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Relabel planar_pyg_dataset.pt with a new defect-atom mapping."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(root_dir, "planar_pyg_dataset.pt"),
        help="Baseline planar dataset to copy geometry/features from.",
    )
    parser.add_argument(
        "--defect-atoms-json",
        type=str,
        required=True,
        help="Alternate laves_defect_atoms_*.json mapping.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pt path (must not overwrite the baseline unless forced).",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Optional JSON stats path (default: <output>_stats.json).",
    )
    parser.add_argument(
        "--simulations-dir",
        type=str,
        default=os.path.join(root_dir, "Laves_Screen_new", "SIMULATIONS"),
        help="Used to read cell matrices for min-image dist_to_defect.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    args = parser.parse_args()

    baseline = os.path.abspath(os.path.join(root_dir, "planar_pyg_dataset.pt"))
    output_abs = os.path.abspath(args.output)
    if output_abs == baseline and not args.force:
        raise SystemExit(
            "Refusing to overwrite the baseline planar_pyg_dataset.pt "
            "(pass --force if you really mean it)."
        )
    if os.path.isfile(args.output) and not args.force:
        raise SystemExit(
            f"Output already exists: {args.output} (pass --force to overwrite)."
        )
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input dataset not found: {args.input}")
    if not os.path.isfile(args.defect_atoms_json):
        raise SystemExit(f"Defect mapping not found: {args.defect_atoms_json}")

    mapping = load_defect_atoms_by_stack(json_path=args.defect_atoms_json)
    print(
        f"Loaded {len(mapping)} stack mappings from {args.defect_atoms_json} "
        f"({sum(1 for v in mapping.values() if v)} non-empty)."
    )

    dataset = torch.load(args.input, weights_only=False)
    print(f"Loaded {len(dataset)} graphs from {args.input}")

    sim_dir = args.simulations_dir if os.path.isdir(args.simulations_dir) else None
    if sim_dir is None:
        print(
            f"  [warn] simulations dir not found ({args.simulations_dir}); "
            "dist_to_defect will use Euclidean distances without PBC."
        )

    relabeled, stats = relabel_dataset(dataset, mapping, simulations_dir=sim_dir)
    stats["source_dataset"] = os.path.abspath(args.input)
    stats["defect_atoms_json"] = os.path.abspath(args.defect_atoms_json)
    stats["node_features"] = [
        "type",
        "per_atom_pe",
        "is_defect",
        "dist_to_defect",
        "cycle3_norm",
        "cycle4_norm",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(relabeled, args.output)
    stats_path = args.stats_output or (
        os.path.splitext(args.output)[0] + "_stats.json"
    )
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Saved {len(relabeled)} graphs -> {args.output}")
    print(f"Saved stats -> {stats_path}")
    print(
        f"Labeled graphs: {stats['graphs_with_defect_labels']}/{stats['num_graphs']}, "
        f"total defect atoms: {stats['total_defect_atoms']}"
    )
    for stack, n_def in sorted(stats["defect_atoms_per_stack"].items()):
        print(f"  {stack!r}: {n_def} defect atoms / graph")


if __name__ == "__main__":
    main()
