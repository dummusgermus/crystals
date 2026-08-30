"""Build k=13 point dataset (edge_k=3, production wiring).

Graph size cutoff_k=13; edge_k=3 matches production. Higher edge_k tiers were
dropped — re-counting cycle34 features on denser wiring is intractable at this
graph size (~250 nodes).

Outputs:
  adv_datasets/cycle34_residual_totals_k13_edge_{e03..e06}.pt
  adv_datasets/point_k13_edge_manifest.json

Local build (jara-ovito + SIMULATIONS/)::

    python build_point_k13_edge_datasets.py

Cluster training uses SKIP_BUILD=1 after syncing .pt files via git.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from adv_graph_maker import DATASET_VARIANTS, augment_cycle_variant
from graph_maker import SHELL_TOL_REL
from k13_edge_datasets_config import (
    ADV_DIR,
    EDGE_K_TIERS,
    FIXED_CUTOFF_K,
    MANIFEST_JSON,
    POINT_FULL_CELL_ATOMS,
    dataset_path,
    stats_path,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
K13_SIZE_SOURCE = os.path.join(ADV_DIR, "cycle34_residual_totals_size_k13.pt")


def _summarize_dataset(
    dataset,
    *,
    tag: str,
    edge_k: int,
    built_from_copy: bool = False,
) -> dict:
    subset_sizes = [int(data.meta.get("subset_size", data.num_nodes)) for data in dataset]
    full_sizes = [
        int(data.meta.get("full_cell_size", POINT_FULL_CELL_ATOMS)) for data in dataset
    ]
    fractions = [ss / max(fs, 1) for ss, fs in zip(subset_sizes, full_sizes)]
    deltas = [float(d.delta_total_eV.view(-1)[0].item()) for d in dataset]
    graph_deltas = [float(d.meta.get("graph_delta_total_eV", float("nan"))) for d in dataset]
    mismatches = [abs(gd - dt) for gd, dt in zip(graph_deltas, deltas) if np.isfinite(gd)]
    cutoff_dists = [float(d.meta.get("cutoff_distance", float("nan"))) for d in dataset]
    edge_counts = [int(data.edge_index.shape[1]) for data in dataset]
    return {
        "tag": tag,
        "cutoff_k": FIXED_CUTOFF_K,
        "edge_k": edge_k,
        "cutoff_mode": "shell",
        "built_from_copy": built_from_copy,
        "path": dataset_path(tag),
        "num_graphs": len(dataset),
        "subset_size_mean": float(np.mean(subset_sizes)),
        "subset_size_median": float(np.median(subset_sizes)),
        "subset_size_min": int(np.min(subset_sizes)),
        "subset_size_max": int(np.max(subset_sizes)),
        "edge_count_mean": float(np.mean(edge_counts)),
        "edge_count_median": float(np.median(edge_counts)),
        "cutoff_distance_mean": float(np.nanmean(cutoff_dists)),
        "cutoff_distance_median": float(np.nanmedian(cutoff_dists)),
        "full_cell_size_mean": float(np.mean(full_sizes)),
        "subset_fraction_actual_mean": float(np.mean(fractions)),
        "subset_fraction_actual_median": float(np.median(fractions)),
        "delta_total_mean_eV": float(np.mean(deltas)),
        "graph_delta_mean_eV": float(np.mean(graph_deltas)),
        "full_graph_delta_mismatch_mean_eV": float(np.mean(mismatches))
        if mismatches
        else 0.0,
        "full_graph_delta_mismatch_median_eV": float(np.median(mismatches))
        if mismatches
        else 0.0,
    }


def _verify_edge_k(dataset, edge_k: int) -> None:
    stored = {int(getattr(d, "edge_k", -1)) for d in dataset[: min(len(dataset), 8)]}
    if stored != {edge_k}:
        raise ValueError(f"Dataset edge_k mismatch: expected {edge_k}, saw {stored}")


def _save_tier(
    dataset,
    *,
    tag: str,
    edge_k: int,
    built_from_copy: bool = False,
) -> dict:
    out = dataset_path(tag)
    os.makedirs(ADV_DIR, exist_ok=True)
    torch.save(dataset, out)
    summary = _summarize_dataset(
        dataset, tag=tag, edge_k=edge_k, built_from_copy=built_from_copy
    )
    with open(stats_path(tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"[{tag}] cutoff_k={FIXED_CUTOFF_K} edge_k={edge_k} -> {out} | "
        f"nodes med={summary['subset_size_median']:.0f} | "
        f"edges med={summary['edge_count_median']:.0f}",
        flush=True,
    )
    return summary


K13_SIZE_SOURCE = os.path.join(ADV_DIR, "cycle34_residual_totals_size_k13.pt")
E03_SOURCE = os.path.join(ADV_DIR, "cycle34_residual_totals_k13_edge_e03.pt")
CYCLE34_FEATURE_DIM = len(DATASET_VARIANTS["cycle34"])


def _shell_threshold(sorted_distances: np.ndarray, k_shells: int) -> float:
    base_dist = sorted_distances[0]
    shell_tol = max(base_dist * SHELL_TOL_REL, 1e-6)
    shell_distances = [float(sorted_distances[0])]
    for dist in sorted_distances[1:]:
        if abs(dist - shell_distances[-1]) > shell_tol:
            shell_distances.append(float(dist))
    cutoff_idx = min(k_shells, len(shell_distances)) - 1
    return shell_distances[cutoff_idx]


def _defect_local_index(data: Data) -> int:
    defect_id = int(getattr(data, "defect_id", -1))
    particle_ids = data.particle_ids.view(-1).tolist()
    for idx, pid in enumerate(particle_ids):
        if int(pid) == defect_id:
            return idx
    raise ValueError(f"Defect id {defect_id} not found in graph particle_ids.")


def _rewire_graph_edges(data: Data, edge_k: int) -> Data:
    """Rewire intra-graph edges at a new edge_k shell; keep nodes/targets fixed."""
    pos = data.pos.detach().cpu().numpy()
    types = data.x[:, 0].detach().cpu().numpy()
    defect_idx = _defect_local_index(data)
    n = int(data.num_nodes)
    dist_matrix = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        neighbor_dists = [float(d) for j, d in enumerate(dist_matrix[i]) if j != i and d > 0.0]
        if not neighbor_dists:
            continue
        sorted_dist = np.sort(np.array(neighbor_dists))
        edge_dist = _shell_threshold(sorted_dist, edge_k)
        for j in range(n):
            if i == j:
                continue
            if dist_matrix[i, j] <= edge_dist:
                src, dst = i, j
                key = (min(src, dst), max(src, dst))
                if key in edge_set:
                    continue
                edge_set.add(key)
                same_type = 1.0 if types[i] == types[j] else 0.0
                incident_defect = 1.0 if (i == defect_idx or j == defect_idx) else 0.0
                edge_index.append([src, dst])
                edge_attr.append([float(dist_matrix[i, j]), same_type, incident_defect])

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 3), dtype=torch.float)

    base_x = data.x[:, : int(data.x.size(-1)) - CYCLE34_FEATURE_DIM]
    new_data = data.clone()
    new_data.x = base_x
    new_data.edge_index = edge_index_tensor
    new_data.edge_attr = edge_attr_tensor
    new_data.edge_k = edge_k
    return new_data


def _rewire_from_e03(dataset, edge_k: int) -> list:
    print(f"Rewiring {len(dataset)} graphs to edge_k={edge_k} from e03 base …", flush=True)
    return [_rewire_graph_edges(data, edge_k) for data in dataset]


def _load_e03_baseline() -> list:
    source = E03_SOURCE if os.path.isfile(E03_SOURCE) else K13_SIZE_SOURCE
    if not os.path.isfile(source):
        raise SystemExit(
            f"Missing e03/k13 baseline at {E03_SOURCE} or {K13_SIZE_SOURCE}"
        )
    print(f"Loading baseline graphs from {source}", flush=True)
    return torch.load(source, weights_only=False)


def build_tier(tag: str, edge_k: int, *, force: bool) -> dict:
    out = dataset_path(tag)
    if os.path.isfile(out) and not force:
        print(f"[{tag}] exists, skipping: {out}")
        with open(stats_path(tag), encoding="utf-8") as fh:
            return json.load(fh)

    if (
        edge_k == 3
        and os.path.isfile(K13_SIZE_SOURCE)
        and not force
    ):
        print(
            f"[{tag}] copying k13 size baseline (edge_k=3) from {K13_SIZE_SOURCE}",
            flush=True,
        )
        shutil.copy2(K13_SIZE_SOURCE, out)
        dataset = torch.load(out, weights_only=False)
        _verify_edge_k(dataset, edge_k)
        return _save_tier(dataset, tag=tag, edge_k=edge_k, built_from_copy=True)

    base = _load_e03_baseline()
    _verify_edge_k(base, 3)
    if edge_k == 3:
        dataset = base
    else:
        rewired = _rewire_from_e03(base, edge_k)
        dataset = augment_cycle_variant(
            rewired,
            variant="cycle34",
            base_feature_dim=int(base[0].x.size(-1)) - CYCLE34_FEATURE_DIM,
        )
    _verify_edge_k(dataset, edge_k)
    return _save_tier(dataset, tag=tag, edge_k=edge_k)


def write_manifest(summaries: Dict[str, dict]) -> None:
    payload = {
        "domain": "point",
        "cutoff_k": FIXED_CUTOFF_K,
        "selection": "edge_k (internal shell wiring within fixed k=13 subgraph)",
        "tiers": [
            {
                "tag": tag,
                "edge_k": ek,
                "dataset": dataset_path(tag),
                "stats": stats_path(tag),
            }
            for tag, ek in EDGE_K_TIERS
        ],
        "summaries": summaries,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Manifest -> {MANIFEST_JSON}", flush=True)


def load_all_summaries() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tag, _ in EDGE_K_TIERS:
        sp = stats_path(tag)
        if os.path.isfile(sp):
            with open(sp, encoding="utf-8") as fh:
                out[tag] = json.load(fh)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build k=13 point datasets with edge_k wiring sweep."
    )
    parser.add_argument(
        "--tier",
        choices=[t for t, _ in EDGE_K_TIERS],
        help="Build one tier only.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tiers: Sequence[Tuple[str, int]] = EDGE_K_TIERS
    if args.tier:
        tiers = [(args.tier, dict(EDGE_K_TIERS)[args.tier])]

    for tag, edge_k in tiers:
        build_tier(tag, edge_k, force=args.force)

    write_manifest(load_all_summaries())
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
