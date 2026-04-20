"""Advanced graph maker with cycle-count node features.

Extends the base pipeline from graph_maker.py by computing per-node
cycle counts (3-, 4-, and 5-cycles) and appending them as z-score
normalised features.  Produces three dataset variants:

  cycle3   – base features + normalised 3-cycle count
  cycle34  – base features + normalised 3- and 4-cycle counts
  cycle345 – base features + normalised 3-, 4-, and 5-cycle counts

Normalisation statistics (mean, std) are persisted as JSON next to each
dataset so they can be applied identically at inference time.

Each :class:`~torch_geometric.data.Data` inherits the per-node atomic
number (``data.z``) and folder tag (``data.folder``) stamped by
:func:`graph_maker.build_pyg_dataset`, so cycle-augmented datasets are
directly usable for the ct-UAE workflow.
"""

import json
import os
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from graph_maker import (
    DEFECT_CUTOFF_K,
    DEFECT_CUTOFF_RADIUS,
    EDGE_CUTOFF_RADIUS,
    EDGE_K,
    build_pyg_dataset,
    save_dataset,
)

CYCLE_LENGTHS = (3, 4, 5)

DATASET_VARIANTS: Dict[str, List[int]] = {
    "cycle3": [0],
    "cycle34": [0, 1],
    "cycle345": [0, 1, 2],
}


# ---------------------------------------------------------------------------
# Cycle counting
# ---------------------------------------------------------------------------

def _edge_index_to_nx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Convert a PyG edge_index tensor to an undirected NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for k in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, k]), int(edge_index[1, k]))
    return G


def _dfs_cycle_count(
    G: nx.Graph,
    start: int,
    current: int,
    path: List[int],
    visited: set,
    counts: np.ndarray,
    max_k: int,
) -> None:
    """Enumerate simple cycles originating at *start* through nodes > start.

    Each cycle of length *k* is discovered in two orientations, so the
    caller must halve the accumulated counts.
    """
    depth = len(path)
    for nbr in G[current]:
        if nbr == start and depth >= 3:
            col = depth - 3
            if col < counts.shape[1]:
                for node in path:
                    counts[node, col] += 1
        elif nbr > start and nbr not in visited and depth < max_k:
            visited.add(nbr)
            path.append(nbr)
            _dfs_cycle_count(G, start, nbr, path, visited, counts, max_k)
            path.pop()
            visited.discard(nbr)


def count_cycles_per_node(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_cycle_len: int = 5,
) -> np.ndarray:
    """Per-node counts of simple 3-, 4- and 5-cycles.

    Returns an array of shape ``[num_nodes, max_cycle_len - 2]``.  Column
    *i* holds the number of simple cycles of length ``i + 3`` passing
    through each node.
    """
    G = _edge_index_to_nx(edge_index, num_nodes)
    n_cols = max_cycle_len - 2
    counts = np.zeros((num_nodes, n_cols), dtype=np.int64)

    for start in range(num_nodes):
        _dfs_cycle_count(G, start, start, [start], {start}, counts, max_k=max_cycle_len)

    counts //= 2  # each cycle found in both orientations
    return counts.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_adv_datasets(
    simulations_dir: str,
    output_dir: str,
    cutoff_k: int = DEFECT_CUTOFF_K,
    edge_k: int = EDGE_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    edge_radius: float = EDGE_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
    variants: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build the base dataset, compute cycle features, and save variants.

    Parameters
    ----------
    variants:
        Subset of :data:`DATASET_VARIANTS` keys to produce; defaults to
        all three (``cycle3``, ``cycle34``, ``cycle345``).

    Returns
    -------
    dict
        Mapping *variant_name → dataset_file_path* for every variant that
        was actually written to disk.
    """
    selected = variants or list(DATASET_VARIANTS.keys())
    unknown = [v for v in selected if v not in DATASET_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {unknown}; choose from {list(DATASET_VARIANTS)}"
        )
    # -- base dataset ----------------------------------------------------------
    print("Building base dataset from simulations …")
    t0 = time.time()
    dataset = build_pyg_dataset(
        simulations_dir=simulations_dir,
        cutoff_k=cutoff_k,
        edge_k=edge_k,
        cutoff_radius=cutoff_radius,
        edge_radius=edge_radius,
        cutoff_mode=cutoff_mode,
    )
    if not dataset:
        raise RuntimeError("No graphs were built – check the simulations directory.")
    print(f"  {len(dataset)} graphs built in {time.time() - t0:.1f}s")

    # -- per-node cycle counts -------------------------------------------------
    print("Computing per-node cycle counts …")
    t0 = time.time()
    all_cycle_feats: List[np.ndarray] = []
    for i, data in enumerate(dataset):
        feats = count_cycles_per_node(data.edge_index, data.num_nodes)
        all_cycle_feats.append(feats)
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed")
    print(f"  Cycle counting finished in {time.time() - t0:.1f}s")

    # -- z-score statistics (computed over every node in the dataset) -----------
    all_cycles = np.concatenate(all_cycle_feats, axis=0)
    mean_all = all_cycles.mean(axis=0)
    std_all = all_cycles.std(axis=0)
    std_all[std_all < 1e-8] = 1.0

    print(f"  Cycle feature means : {mean_all}")
    print(f"  Cycle feature stds  : {std_all}")

    # -- save each variant -----------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    for variant in selected:
        cols = DATASET_VARIANTS[variant]
        v_mean = mean_all[cols]
        v_std = std_all[cols]

        variant_ds: List[Data] = []
        for data, cyc in zip(dataset, all_cycle_feats):
            normed = (cyc[:, cols] - v_mean) / v_std
            new_data = data.clone()
            new_data.x = torch.cat(
                [data.x, torch.tensor(normed, dtype=torch.float)], dim=-1
            )
            variant_ds.append(new_data)

        ds_path = os.path.join(output_dir, f"{variant}_dataset.pt")
        save_dataset(variant_ds, ds_path)

        stats = {
            "cycle_lengths": [CYCLE_LENGTHS[c] for c in cols],
            "mean": v_mean.tolist(),
            "std": v_std.tolist(),
            "num_graphs": len(variant_ds),
            "base_feature_dim": int(dataset[0].x.size(-1)),
        }
        stats_path = os.path.join(output_dir, f"{variant}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        saved[variant] = ds_path
        print(f"  [{variant}] {len(variant_ds)} graphs → {ds_path}")
        print(f"  [{variant}] stats → {stats_path}")

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Build cycle-augmented PyG datasets. Defaults write to "
            "'adv_datasets_uae/' so the original 'adv_datasets/' files are "
            "preserved; the cycle features are identical, only the per-node "
            "Data.z / Data.folder tags are new."
        )
    )
    parser.add_argument(
        "--simulations-dir",
        type=str,
        default=os.path.join(root_dir, "SIMULATIONS"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(root_dir, "adv_datasets_uae"),
        help="Directory for the cycle-augmented datasets and stats JSONs.",
    )
    parser.add_argument("--cutoff-k", type=int, default=DEFECT_CUTOFF_K)
    parser.add_argument("--edge-k", type=int, default=EDGE_K)
    parser.add_argument("--cutoff-radius", type=float, default=DEFECT_CUTOFF_RADIUS)
    parser.add_argument("--edge-radius", type=float, default=EDGE_CUTOFF_RADIUS)
    parser.add_argument("--cutoff-mode", choices=["shell", "radius"], default="shell")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DATASET_VARIANTS.keys()),
        choices=list(DATASET_VARIANTS.keys()),
        help="Which cycle variants to write (default: all).",
    )
    args = parser.parse_args()

    build_adv_datasets(
        simulations_dir=args.simulations_dir,
        output_dir=args.output_dir,
        cutoff_k=args.cutoff_k,
        edge_k=args.edge_k,
        cutoff_radius=args.cutoff_radius,
        edge_radius=args.edge_radius,
        cutoff_mode=args.cutoff_mode,
        variants=args.variants,
    )
    print("All done.")
