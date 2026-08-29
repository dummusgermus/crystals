"""Build datasets with full-cell total residual as the graph-level target.

Keeps all node/edge features unchanged. Per-atom residuals are preserved on
``y_atom_residual``; ``y`` becomes a single scalar per graph:

    y = delta_total_eV = PE_true_total - PE_initial_total

Outputs:

* ``adv_datasets/cycle34_graph_total_target_dataset.pt``
* ``planar_pyg_dataset_graph_total_target_c14c15.pt``

Source: existing ``*_totals_dataset.pt`` if present, otherwise rebuild totals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

POINT_TOTALS = os.path.join(ROOT, "adv_datasets", "cycle34_residual_totals_dataset.pt")
PLANAR_TOTALS = os.path.join(
    ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt"
)

POINT_OUTPUT = os.path.join(ROOT, "adv_datasets", "cycle34_graph_total_target_dataset.pt")
POINT_STATS = os.path.join(ROOT, "adv_datasets", "cycle34_graph_total_target_stats.json")
PLANAR_OUTPUT = os.path.join(
    ROOT, "planar_pyg_dataset_graph_total_target_c14c15.pt"
)
PLANAR_STATS = os.path.join(
    ROOT, "planar_pyg_dataset_graph_total_target_c14c15_stats.json"
)


def _get_delta_total(data) -> torch.Tensor:
    if hasattr(data, "delta_total_eV") and data.delta_total_eV is not None:
        return data.delta_total_eV.view(1, 1).clone().float()
    meta = getattr(data, "meta", None) or {}
    if "delta_total_eV" in meta:
        return torch.tensor([[float(meta["delta_total_eV"])]], dtype=torch.float)
    raise ValueError("Graph missing delta_total_eV")


def convert_to_graph_total_target(dataset) -> list:
    out = []
    for data in dataset:
        d = data.clone()
        if hasattr(d, "y") and d.y is not None and d.y.numel() > 1:
            d.y_atom_residual = d.y.clone()
        d.y = _get_delta_total(d)
        if not hasattr(d, "delta_total_eV") or d.delta_total_eV is None:
            d.delta_total_eV = d.y.view(1).clone()
        d.target_mode = "graph_total_residual"
        out.append(d)
    return out


def _summarize(dataset, path: str) -> dict:
    deltas = [float(d.y.view(-1)[0].item()) for d in dataset]
    abs_d = [abs(x) for x in deltas]
    return {
        "path": path,
        "num_graphs": len(dataset),
        "target": "delta_total_eV (full-cell net residual)",
        "delta_mean_eV": float(sum(deltas) / max(len(deltas), 1)),
        "delta_std_eV": float(torch.tensor(deltas).std(unbiased=False).item())
        if deltas
        else 0.0,
        "abs_delta_mean_eV": float(sum(abs_d) / max(len(abs_d), 1)),
        "n_near_zero": int(sum(1 for x in abs_d if x < 1e-6)),
    }


def _ensure_totals(point_only: bool, planar_only: bool, force: bool) -> None:
    need_point = not planar_only
    need_planar = not point_only
    missing = []
    if need_point and not os.path.isfile(POINT_TOTALS):
        missing.append(POINT_TOTALS)
    if need_planar and not os.path.isfile(PLANAR_TOTALS):
        missing.append(PLANAR_TOTALS)
    if not missing and not force:
        return
    cmd = [sys.executable, os.path.join(ROOT, "build_residual_datasets_with_totals.py")]
    if force:
        cmd.append("--force")
    if point_only:
        cmd.append("--point-only")
    if planar_only:
        cmd.append("--planar-only")
    print("Building totals datasets (prerequisite) …")
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_point(force: bool = False) -> None:
    if os.path.isfile(POINT_OUTPUT) and not force:
        print(f"Exists, skipping: {POINT_OUTPUT}")
        return
    _ensure_totals(point_only=True, planar_only=False, force=force)
    dataset = torch.load(POINT_TOTALS, weights_only=False)
    converted = convert_to_graph_total_target(dataset)
    torch.save(converted, POINT_OUTPUT)
    summary = _summarize(converted, POINT_OUTPUT)
    with open(POINT_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {POINT_OUTPUT} ({summary['num_graphs']} graphs)")


def build_planar(force: bool = False) -> None:
    if os.path.isfile(PLANAR_OUTPUT) and not force:
        print(f"Exists, skipping: {PLANAR_OUTPUT}")
        return
    _ensure_totals(point_only=False, planar_only=True, force=force)
    dataset = torch.load(PLANAR_TOTALS, weights_only=False)
    converted = convert_to_graph_total_target(dataset)
    torch.save(converted, PLANAR_OUTPUT)
    summary = _summarize(converted, PLANAR_OUTPUT)
    with open(PLANAR_STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {PLANAR_OUTPUT} ({summary['num_graphs']} graphs)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graph-level total-residual target datasets."
    )
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.planar_only:
        build_point(force=args.force)
    if not args.point_only:
        build_planar(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
