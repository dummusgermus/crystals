"""Build residual datasets with full-cell energy totals attached to each graph.

Outputs (separate from legacy delivery datasets):

* ``adv_datasets/cycle34_residual_totals_dataset.pt``
* ``planar_pyg_dataset_residual_c14c15_totals.pt``

Each ``Data`` object has ``delta_total_eV`` (shape ``[1]``) and ``meta`` fields
``pe_initial_total_eV``, ``pe_true_total_eV``, ``delta_total_eV``,
``graph_delta_total_eV``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from adv_graph_maker import build_adv_datasets
from planar_graph_maker import build_planar_dataset_with_cycles

ROOT = os.path.dirname(os.path.abspath(__file__))

POINT_OUTPUT = os.path.join(ROOT, "adv_datasets", "cycle34_residual_totals_dataset.pt")
POINT_STATS = os.path.join(ROOT, "adv_datasets", "cycle34_residual_totals_stats.json")
PLANAR_OUTPUT = os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt")
PLANAR_STATS = os.path.join(
    ROOT, "planar_pyg_dataset_residual_c14c15_totals_stats.json"
)
DEFECT_ATOMS_JSON = os.path.join(ROOT, "laves_defect_atoms_c14c15.json")
INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")
SIMULATIONS_DIR = os.path.join(ROOT, "SIMULATIONS")


def _summarize_dataset(path: str) -> dict:
    dataset = torch.load(path, weights_only=False)
    deltas = [float(d.delta_total_eV.view(-1)[0].item()) for d in dataset]
    graph_deltas = [
        float(d.meta.get("graph_delta_total_eV", float("nan"))) for d in dataset
    ]
    return {
        "path": path,
        "num_graphs": len(dataset),
        "delta_total_mean_eV": float(sum(deltas) / max(len(deltas), 1)),
        "delta_total_std_eV": float(torch.tensor(deltas).std(unbiased=False).item())
        if deltas
        else 0.0,
        "graph_delta_mean_eV": float(sum(graph_deltas) / max(len(graph_deltas), 1)),
    }


def build_point(force: bool = False) -> None:
    if os.path.isfile(POINT_OUTPUT) and not force:
        print(f"Exists, skipping: {POINT_OUTPUT}")
        return
    print("=== Point residual dataset with totals ===")
    build_adv_datasets(
        simulations_dir=SIMULATIONS_DIR,
        output_dir=os.path.join(ROOT, "adv_datasets"),
        variants=["cycle34"],
        target_mode="residual",
    )
    legacy = os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt")
    if not os.path.isfile(legacy):
        raise SystemExit(f"Expected build output missing: {legacy}")
    dataset = torch.load(legacy, weights_only=False)
    missing = sum(1 for d in dataset if not hasattr(d, "delta_total_eV"))
    if missing:
        raise SystemExit(
            f"{missing} graphs missing delta_total_eV; rebuild graph_maker first."
        )
    torch.save(dataset, POINT_OUTPUT)
    summary = _summarize_dataset(POINT_OUTPUT)
    with open(POINT_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {POINT_OUTPUT} ({summary['num_graphs']} graphs)")


def build_planar(force: bool = False) -> None:
    if os.path.isfile(PLANAR_OUTPUT) and not force:
        print(f"Exists, skipping: {PLANAR_OUTPUT}")
        return
    print("=== Planar residual dataset with totals ===")
    build_planar_dataset_with_cycles(
        initial_simulations_dir=INITIAL_DIR,
        relaxed_simulations_dir=RELAXED_DIR,
        output_path=PLANAR_OUTPUT,
        stats_path=PLANAR_STATS,
        defect_atoms_json=DEFECT_ATOMS_JSON,
        target_mode="residual",
        require_initial_pe=True,
    )
    summary = _summarize_dataset(PLANAR_OUTPUT)
    with open(PLANAR_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {PLANAR_OUTPUT} ({summary['num_graphs']} graphs)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build residual datasets with delta_total_eV on each graph."
    )
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_point_flag = not args.planar_only
    build_planar_flag = not args.point_only

    if build_point_flag:
        build_point(force=args.force)
    if build_planar_flag:
        build_planar(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
