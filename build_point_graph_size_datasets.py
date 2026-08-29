"""Build point-defect datasets at increasing shell cutoffs (fast, same as production).

Uses the existing ``cutoff_k`` shell parameter from ``graph_maker.py`` (default k=8).
Each tier adds one coordination shell around the defect — graphs stay ~100–300 atoms,
not thousands.

Tiers: k08 (baseline) … k13, plus k22 (large shell stress test)

Outputs (git-tracked for cluster sync):
  adv_datasets/cycle34_residual_totals_size_{tag}.pt
  adv_datasets/cycle34_residual_totals_size_{tag}_stats.json
  adv_datasets/point_graph_size_manifest.json

Requires SIMULATIONS/ locally. Use the jara-ovito conda env::

    conda activate jara-ovito
    python build_point_graph_size_datasets.py

Cluster training uses SKIP_BUILD=1 (datasets synced via git).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from adv_graph_maker import build_adv_datasets
from graph_maker import DEFECT_CUTOFF_K
from graph_size_datasets_config import (
    ADV_DIR,
    GRAPH_SIZE_TIERS,
    MANIFEST_JSON,
    POINT_FULL_CELL_ATOMS,
    dataset_path,
    stats_path,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SIMULATIONS_DIR = os.path.join(ROOT, "SIMULATIONS")
BASELINE_SOURCE = os.path.join(ADV_DIR, "cycle34_residual_totals_dataset.pt")

def _summarize_dataset(
    dataset, *, tag: str, cutoff_k: int, built_from_copy: bool = False
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
    return {
        "tag": tag,
        "cutoff_k": cutoff_k,
        "cutoff_mode": "shell",
        "built_from_copy": built_from_copy,
        "path": dataset_path(tag),
        "num_graphs": len(dataset),
        "subset_size_mean": float(np.mean(subset_sizes)),
        "subset_size_median": float(np.median(subset_sizes)),
        "subset_size_min": int(np.min(subset_sizes)),
        "subset_size_max": int(np.max(subset_sizes)),
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


def _enrich_point_meta(dataset) -> None:
    for data in dataset:
        meta = dict(data.meta)
        n_sub = int(meta.get("subset_size", data.num_nodes))
        meta["full_cell_size"] = int(meta.get("full_cell_size", POINT_FULL_CELL_ATOMS))
        meta["subset_size"] = n_sub
        meta["subset_fraction_actual"] = n_sub / max(meta["full_cell_size"], 1)
        data.meta = meta


def _save_tier(dataset, *, tag: str, cutoff_k: int, built_from_copy: bool = False) -> dict:
    out = dataset_path(tag)
    os.makedirs(ADV_DIR, exist_ok=True)
    torch.save(dataset, out)
    summary = _summarize_dataset(
        dataset, tag=tag, cutoff_k=cutoff_k, built_from_copy=built_from_copy
    )
    with open(stats_path(tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"[{tag}] k={cutoff_k} -> {out} | "
        f"nodes median={summary['subset_size_median']:.0f} "
        f"({summary['subset_fraction_actual_median'] * 100:.1f}% cell) | "
        f"cutoff dist med={summary['cutoff_distance_median']:.2f} A",
        flush=True,
    )
    return summary


def build_tier(tag: str, cutoff_k: int, *, force: bool) -> dict:
    out = dataset_path(tag)
    if os.path.isfile(out) and not force:
        print(f"[{tag}] exists, skipping: {out}")
        with open(stats_path(tag), encoding="utf-8") as fh:
            return json.load(fh)

    if tag == "k08" and cutoff_k == DEFECT_CUTOFF_K and os.path.isfile(BASELINE_SOURCE):
        print(f"[{tag}] copying production baseline from {BASELINE_SOURCE}", flush=True)
        shutil.copy2(BASELINE_SOURCE, out)
        dataset = torch.load(out, weights_only=False)
        _enrich_point_meta(dataset)
        return _save_tier(dataset, tag=tag, cutoff_k=cutoff_k, built_from_copy=True)

    if not os.path.isdir(SIMULATIONS_DIR):
        raise SystemExit(f"SIMULATIONS/ not found at {SIMULATIONS_DIR}")

    build_dir = os.path.join(ADV_DIR, f"_build_size_{tag}")
    os.makedirs(build_dir, exist_ok=True)
    print(f"[{tag}] building cutoff_k={cutoff_k} (shell) …", flush=True)
    saved = build_adv_datasets(
        simulations_dir=SIMULATIONS_DIR,
        output_dir=build_dir,
        cutoff_k=cutoff_k,
        cutoff_mode="shell",
        variants=["cycle34"],
        target_mode="residual",
    )
    dataset = torch.load(saved["cycle34"], weights_only=False)
    return _save_tier(dataset, tag=tag, cutoff_k=cutoff_k)


def write_manifest(summaries: Dict[str, dict]) -> None:
    payload = {
        "domain": "point",
        "selection": "shell cutoff_k (coordination shells from defect)",
        "tiers": [
            {
                "tag": tag,
                "cutoff_k": k,
                "dataset": dataset_path(tag),
                "stats": stats_path(tag),
            }
            for tag, k in GRAPH_SIZE_TIERS
        ],
        "summaries": summaries,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Manifest -> {MANIFEST_JSON}", flush=True)


def load_all_summaries() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tag, _ in GRAPH_SIZE_TIERS:
        sp = stats_path(tag)
        if os.path.isfile(sp):
            with open(sp, encoding="utf-8") as fh:
                out[tag] = json.load(fh)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build point graph-size datasets via shell cutoff_k sweep."
    )
    parser.add_argument(
        "--tier",
        choices=[t for t, _ in GRAPH_SIZE_TIERS],
        help="Build one tier only.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tiers: Sequence[Tuple[str, int]] = GRAPH_SIZE_TIERS
    if args.tier:
        tiers = [(args.tier, dict(GRAPH_SIZE_TIERS)[args.tier])]

    for tag, cutoff_k in tiers:
        build_tier(tag, cutoff_k, force=args.force)

    write_manifest(load_all_summaries())
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
