"""CGCNN global-v2 sweep over point graph size (baseline + 5 fraction tiers).

Tracks per-atom MAE and net-energy metrics on both graph and full-cell targets.

Example::

    python build_point_graph_fraction_datasets.py
    python train_cgcnn_graph_fraction_sweep.py --skip-build --epochs 300 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from build_point_graph_fraction_datasets import (
    GRAPH_FRACTION_TIERS,
    MANIFEST_JSON,
    dataset_path,
)
from train_cgcnn_extensive_comparison import DEFAULT_CONFIG, _train_one
from train_cgcnn_global_v2 import default_total_loss_config

ROOT = os.path.dirname(os.path.abspath(__file__))

SWEEP_JSON = os.path.join(ROOT, "cgcnn_graph_fraction_sweep_curves.json")
SWEEP_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_graph_fraction_sweep_summary.json")
SWEEP_PLOT_PREFIX = os.path.join(ROOT, "cgcnn_graph_fraction_sweep")

POINT_LAMBDA = 0.01
DEFAULT_EPOCHS = 300


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_key(tag: str) -> str:
    return f"point_graph_frac_{tag}"


def _load_manifest() -> dict:
    if not os.path.isfile(MANIFEST_JSON):
        raise SystemExit(
            f"Missing {MANIFEST_JSON}. Run build_point_graph_fraction_datasets.py first."
        )
    with open(MANIFEST_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _tier_dataset_paths() -> List[Tuple[str, Optional[float], str]]:
    rows: List[Tuple[str, Optional[float], str]] = []
    for tag, frac in GRAPH_FRACTION_TIERS:
        path = dataset_path(tag)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for tier {tag}: {path}")
        rows.append((tag, frac, path))
    return rows


def _build_summary(runs: Dict[str, Dict]) -> dict:
    rows = []
    for tag, frac in GRAPH_FRACTION_TIERS:
        key = _run_key(tag)
        run = runs.get(key)
        if not run:
            continue
        rows.append(
            {
                "tag": tag,
                "subset_fraction_target": frac,
                "run_key": key,
                "dataset_path": run.get("dataset_path"),
                "subset_fraction_actual_median": run.get("subset_fraction_actual_median"),
                "subset_size_median": run.get("subset_size_median"),
                "final_test_mae": run["final_test_mae"],
                "final_test_r_tot_median": run["final_test_r_tot_median"],
                "final_test_r_tot_full_median": run.get(
                    "final_test_r_tot_full_median", float("nan")
                ),
                "final_test_abs_total_err_eV": run["final_test_abs_total_err_eV"],
                "final_test_abs_total_err_full_eV": run.get(
                    "final_test_abs_total_err_full_eV", float("nan")
                ),
                "best_val_score": run["best_val_score"],
            }
        )

    best_graph = min(rows, key=lambda r: r["final_test_r_tot_median"]) if rows else None
    best_full = (
        min(rows, key=lambda r: r["final_test_r_tot_full_median"]) if rows else None
    )
    return {
        "domain": "point",
        "model": "cgcnn",
        "loss": "global_v2",
        "lambda_tot": POINT_LAMBDA,
        "checkpoint_metric": "r_tot_graph",
        "tiers": rows,
        "best_by_test_r_tot_graph": best_graph,
        "best_by_test_r_tot_full": best_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGCNN sweep over point graph fractions (global v2)."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--tier",
        choices=[t for t, _ in GRAPH_FRACTION_TIERS],
        help="Train a single tier only.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default=SWEEP_JSON)
    parser.add_argument("--summary-json", default=SWEEP_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=SWEEP_PLOT_PREFIX)
    args = parser.parse_args()

    if not args.skip_build:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "build_point_graph_fraction_datasets.py"),
        ]
        if args.force_build:
            cmd.append("--force")
        if args.tier:
            cmd.extend(["--tier", args.tier])
        print("Building graph-fraction datasets …", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    if args.build_only:
        print("Build-only done.")
        return

    manifest = _load_manifest()
    summaries_by_tag = manifest.get("summaries", {})
    if not isinstance(summaries_by_tag, dict):
        summaries_by_tag = {}

    tier_rows = _tier_dataset_paths()
    if args.tier:
        tier_rows = [row for row in tier_rows if row[0] == args.tier]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs={args.epochs}, lambda={POINT_LAMBDA}, tiers={[t[0] for t in tier_rows]}")

    payload: dict = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
    runs: Dict[str, Dict] = dict(payload.get("runs", {}))

    total_cfg = default_total_loss_config(POINT_LAMBDA)

    for tag, frac, ds_path in tier_rows:
        run_key = _run_key(tag)
        if run_key in runs:
            print(f"[skip] already trained: {run_key}")
            continue
        dataset = torch.load(ds_path, weights_only=False)
        if not hasattr(dataset[0], "delta_total_eV"):
            raise SystemExit(f"{ds_path} lacks delta_total_eV")

        tier_stats = summaries_by_tag.get(tag, {})
        print(f"\n>>> {run_key} | fraction={frac} | {ds_path}")
        runs[run_key] = _train_one(
            domain="point",
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            seed=args.seed,
            config=DEFAULT_CONFIG,
            use_total_loss=True,
            lambda_tot=POINT_LAMBDA,
            run_key=run_key,
            checkpoint_metric="r_tot",
            total_loss_config=total_cfg,
            legacy_total_loss=False,
            extra_total_eval_modes=("full",),
        )
        runs[run_key]["dataset_path"] = ds_path
        runs[run_key]["graph_fraction_tag"] = tag
        runs[run_key]["subset_fraction_target"] = frac
        runs[run_key]["subset_fraction_actual_median"] = tier_stats.get(
            "subset_fraction_actual_median"
        )
        runs[run_key]["subset_size_median"] = tier_stats.get("subset_size_median")
        _save_json(
            args.output_json,
            {
                "version": "cgcnn_graph_fraction_sweep",
                "domain": "point",
                "epochs": args.epochs,
                "seed": args.seed,
                "lambda_tot": POINT_LAMBDA,
                "manifest": MANIFEST_JSON,
                "runs": runs,
            },
        )

    summary = _build_summary(runs)
    payload = {
        "version": "cgcnn_graph_fraction_sweep",
        "domain": "point",
        "epochs": args.epochs,
        "seed": args.seed,
        "lambda_tot": POINT_LAMBDA,
        "manifest": MANIFEST_JSON,
        "runs": runs,
        "summary": summary,
    }
    _save_json(args.output_json, payload)
    _save_json(args.summary_json, summary)
    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    print(json.dumps(summary, indent=2))

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_graph_fraction_sweep.py")
        subprocess.run(
            [
                sys.executable,
                plot_script,
                "--input",
                args.output_json,
                "--output-prefix",
                args.plot_prefix,
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
