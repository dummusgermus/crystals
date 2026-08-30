"""Build delivery datasets for the k13 global-v2 production pipeline.

Point (k=13, edge_k=3):
  adv_datasets/cycle34_residual_totals_k13_dataset.pt   (training)
  adv_datasets/cycle34_residual_k13_dataset.pt          (inference/export)

Planar (unchanged):
  planar_pyg_dataset_residual_c14c15_totals.pt
  planar_pyg_dataset_residual_c14c15.pt

Local build (jara-ovito + SIMULATIONS/ + Laves_* dirs)::

    python build_delivery_datasets.py

Cluster: build locally, rsync .pt files, use SKIP_BUILD=1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Optional

import torch

from adv_graph_maker import build_adv_datasets
from delivery_global_v2 import (
    EXPORT_DATASETS,
    POINT_CUTOFF_K,
    POINT_EDGE_K,
    TOTALS_DATASETS,
)
from k13_edge_datasets_config import dataset_path as k13_edge_dataset_path
from planar_graph_maker import build_planar_dataset_with_cycles

ROOT = os.path.dirname(os.path.abspath(__file__))
SIMULATIONS_DIR = os.path.join(ROOT, "SIMULATIONS")
DEFECT_ATOMS_JSON = os.path.join(ROOT, "laves_defect_atoms_c14c15.json")
INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")

POINT_TOTALS = TOTALS_DATASETS["point"]
POINT_EXPORT = EXPORT_DATASETS["point"]
POINT_TOTALS_STATS = os.path.join(
    ROOT, "adv_datasets", "cycle34_residual_totals_k13_stats.json"
)
POINT_EXPORT_STATS = os.path.join(
    ROOT, "adv_datasets", "cycle34_residual_k13_stats.json"
)
PLANAR_TOTALS = TOTALS_DATASETS["planar"]
PLANAR_EXPORT = EXPORT_DATASETS["planar"]
PLANAR_TOTALS_STATS = os.path.join(
    ROOT, "planar_pyg_dataset_residual_c14c15_totals_stats.json"
)
PLANAR_EXPORT_STATS = os.path.join(
    ROOT, "planar_pyg_dataset_residual_c14c15_stats.json"
)

K13_EDGE_E03 = k13_edge_dataset_path("e03")
K13_SIZE = os.path.join(ROOT, "adv_datasets", "cycle34_residual_totals_size_k13.pt")


def _verify_point_dataset(dataset, *, path: str) -> None:
    if not dataset:
        raise RuntimeError(f"Empty dataset: {path}")
    g = dataset[0]
    if not hasattr(g, "particle_ids"):
        raise RuntimeError(f"{path} missing particle_ids (needed for inference)")
    if not hasattr(g, "delta_total_eV"):
        raise RuntimeError(f"{path} missing delta_total_eV")
    cutoff = int(getattr(g, "cutoff_k", -1))
    edge_k = int(getattr(g, "edge_k", -1))
    if cutoff != POINT_CUTOFF_K or edge_k != POINT_EDGE_K:
        raise RuntimeError(
            f"{path} expected cutoff_k={POINT_CUTOFF_K} edge_k={POINT_EDGE_K}, "
            f"got cutoff_k={cutoff} edge_k={edge_k}"
        )


def _summarize(path: str) -> dict:
    dataset = torch.load(path, weights_only=False)
    nodes = [int(d.num_nodes) for d in dataset]
    edges = [int(d.edge_index.shape[1]) for d in dataset]
    return {
        "path": path,
        "num_graphs": len(dataset),
        "nodes_median": float(sorted(nodes)[len(nodes) // 2]),
        "edges_median": float(sorted(edges)[len(edges) // 2]),
    }


def _save_point_pair(dataset, *, built_from: str) -> None:
    os.makedirs(os.path.dirname(POINT_TOTALS), exist_ok=True)
    _verify_point_dataset(dataset, path=built_from)
    torch.save(dataset, POINT_TOTALS)
    torch.save(dataset, POINT_EXPORT)
    for out, stats_path in (
        (POINT_TOTALS, POINT_TOTALS_STATS),
        (POINT_EXPORT, POINT_EXPORT_STATS),
    ):
        summary = _summarize(out)
        summary["built_from"] = built_from
        summary["cutoff_k"] = POINT_CUTOFF_K
        summary["edge_k"] = POINT_EDGE_K
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print(
        f"[point] k={POINT_CUTOFF_K} edge_k={POINT_EDGE_K} "
        f"({len(dataset)} graphs) -> {POINT_TOTALS}",
        flush=True,
    )


def build_point(force: bool = False) -> None:
    if (
        os.path.isfile(POINT_TOTALS)
        and os.path.isfile(POINT_EXPORT)
        and not force
    ):
        print(f"[point] exists, skipping: {POINT_TOTALS}")
        return

    for source in (K13_EDGE_E03, K13_SIZE):
        if os.path.isfile(source):
            print(f"[point] copying from {source}", flush=True)
            dataset = torch.load(source, weights_only=False)
            _save_point_pair(dataset, built_from=source)
            return

    if not os.path.isdir(SIMULATIONS_DIR):
        raise SystemExit(f"SIMULATIONS/ not found at {SIMULATIONS_DIR}")
    build_dir = os.path.join(ROOT, "adv_datasets", "_build_delivery_k13")
    os.makedirs(build_dir, exist_ok=True)
    print(
        f"[point] building cutoff_k={POINT_CUTOFF_K} edge_k={POINT_EDGE_K} …",
        flush=True,
    )
    saved = build_adv_datasets(
        simulations_dir=SIMULATIONS_DIR,
        output_dir=build_dir,
        cutoff_k=POINT_CUTOFF_K,
        edge_k=POINT_EDGE_K,
        cutoff_mode="shell",
        variants=["cycle34"],
        target_mode="residual",
    )
    dataset = torch.load(saved["cycle34"], weights_only=False)
    _save_point_pair(dataset, built_from=build_dir)


def build_planar(force: bool = False) -> None:
    if os.path.isfile(PLANAR_TOTALS) and os.path.isfile(PLANAR_EXPORT) and not force:
        print(f"[planar] exists, skipping: {PLANAR_TOTALS}")
        return
    print("[planar] building residual datasets …", flush=True)
    build_planar_dataset_with_cycles(
        initial_simulations_dir=INITIAL_DIR,
        relaxed_simulations_dir=RELAXED_DIR,
        output_path=PLANAR_TOTALS,
        stats_path=PLANAR_TOTALS_STATS,
        defect_atoms_json=DEFECT_ATOMS_JSON,
        target_mode="residual",
        require_initial_pe=True,
    )
    if not os.path.isfile(PLANAR_EXPORT):
        shutil.copy2(PLANAR_TOTALS, PLANAR_EXPORT)
    summary = _summarize(PLANAR_TOTALS)
    with open(PLANAR_TOTALS_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(PLANAR_EXPORT_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[planar] -> {PLANAR_TOTALS} ({summary['num_graphs']} graphs)", flush=True)


def datasets_ready() -> bool:
    return all(os.path.isfile(p) for p in (POINT_TOTALS, POINT_EXPORT, PLANAR_TOTALS, PLANAR_EXPORT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build delivery k13 + planar datasets.")
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.planar_only:
        build_point(force=args.force)
    if not args.point_only:
        build_planar(force=args.force)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
