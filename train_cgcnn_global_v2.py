"""Quick sanity check: atom-only vs improved global loss (single lambda).

Improvements over the failed lambda sweep:
  * train/eval on graph_delta_total_eV (subgraph-consistent target)
  * Huber loss on scaled relative net-energy error
  * loss balancing so lambda is meaningful vs per-atom MAE
  * down-weight outlier graphs (large |delta| or full/graph mismatch)

Trains baseline + one global run per domain (default lambda=0.01, 100 epochs).

Example::

    python train_cgcnn_global_v2.py --skip-build --epochs 100 --plot
    python train_cgcnn_global_v2.py --domain planar --lambda-tot 0.01 --skip-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict

import torch

from train_cgcnn_extensive_comparison import (
    DATASETS,
    DEFAULT_CONFIG,
    _train_one,
    build_datasets,
)
from train_single import TotalLossConfig

ROOT = os.path.dirname(os.path.abspath(__file__))

V2_JSON = os.path.join(ROOT, "cgcnn_global_v2_curves.json")
V2_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_global_v2_summary.json")
V2_PLOT = os.path.join(ROOT, "cgcnn_global_v2_curves.png")

DEFAULT_LAMBDA = 0.01
DEFAULT_EPOCHS = 100


def default_total_loss_config(lambda_tot: float) -> TotalLossConfig:
    return TotalLossConfig(
        target_mode="graph",
        loss_type="huber",
        scale_eps=0.05,
        huber_delta=1.0,
        lambda_tot=lambda_tot,
        balance_losses=True,
        outlier_max_delta_eV=50.0,
        outlier_max_mismatch_eV=5.0,
    )


def _build_summary(runs: Dict[str, Dict]) -> dict:
    summary = {"baseline_vs_global_v2": {}}
    for domain in ("point", "planar"):
        baseline_key = f"{domain}_atom_only"
        global_key = f"{domain}_global_v2"
        baseline = runs.get(baseline_key)
        global_run = runs.get(global_key)
        if not baseline or not global_run:
            continue
        summary["baseline_vs_global_v2"][domain] = {
            "baseline_run_key": baseline_key,
            "global_run_key": global_key,
            "lambda_tot": global_run.get("lambda_tot"),
            "baseline": {
                "final_test_mae": baseline["final_test_mae"],
                "final_test_r_tot_median": baseline["final_test_r_tot_median"],
                "final_test_abs_total_err_eV": baseline["final_test_abs_total_err_eV"],
            },
            "global_v2": {
                "final_test_mae": global_run["final_test_mae"],
                "final_test_r_tot_median": global_run["final_test_r_tot_median"],
                "final_test_abs_total_err_eV": global_run["final_test_abs_total_err_eV"],
            },
            "delta_test_r_tot_median_pct": (
                global_run["final_test_r_tot_median"]
                - baseline["final_test_r_tot_median"]
            ),
            "delta_test_mae_eV": (
                global_run["final_test_mae"] - baseline["final_test_mae"]
            ),
            "delta_test_abs_total_err_eV": (
                global_run["final_test_abs_total_err_eV"]
                - baseline["final_test_abs_total_err_eV"]
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGCNN atom-only vs improved global loss (single lambda sanity check)."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--lambda-tot", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument(
        "--domain",
        choices=["point", "planar", "both"],
        default="both",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output-json", default=V2_JSON)
    parser.add_argument("--summary-json", default=V2_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-output", default=V2_PLOT)
    args = parser.parse_args()

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    domains = ("point", "planar") if args.domain == "both" else (args.domain,)
    for domain in domains:
        if not os.path.isfile(DATASETS[domain]):
            raise SystemExit(f"Missing dataset: {DATASETS[domain]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Global v2 lambda_tot={args.lambda_tot:g}, epochs={args.epochs}")

    total_cfg = default_total_loss_config(args.lambda_tot)
    runs: Dict[str, Dict] = {}

    for domain in domains:
        dataset_path = DATASETS[domain]
        dataset = torch.load(dataset_path, weights_only=False)
        if not hasattr(dataset[0], "delta_total_eV"):
            raise SystemExit(f"{dataset_path} lacks delta_total_eV; rebuild datasets.")

        baseline_key = f"{domain}_atom_only"
        if not args.skip_baseline:
            print(f"\n>>> Baseline {baseline_key}")
            runs[baseline_key] = _train_one(
                domain=domain,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=DEFAULT_CONFIG,
                use_total_loss=False,
                lambda_tot=0.0,
                run_key=baseline_key,
                checkpoint_metric="mae",
            )
            runs[baseline_key]["dataset_path"] = dataset_path

        global_key = f"{domain}_global_v2"
        if not args.skip_global:
            print(f"\n>>> Global v2 {global_key}")
            runs[global_key] = _train_one(
                domain=domain,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=DEFAULT_CONFIG,
                use_total_loss=True,
                lambda_tot=args.lambda_tot,
                run_key=global_key,
                checkpoint_metric="mae",
                total_loss_config=total_cfg,
                legacy_total_loss=False,
            )
            runs[global_key]["dataset_path"] = dataset_path

    summary = _build_summary(runs)
    payload = {
        "version": "global_v2",
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "split": "grouped 70/15/15",
        "lambda_tot": args.lambda_tot,
        "total_loss_config": total_cfg.__dict__,
        "config": DEFAULT_CONFIG,
        "runs": runs,
        "summary": summary,
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(args.summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    print(json.dumps(summary, indent=2))

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_extensive_comparison.py")
        subprocess.run(
            [
                sys.executable,
                plot_script,
                "--input",
                args.output_json,
                "--output",
                args.plot_output,
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
