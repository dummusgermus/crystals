"""Lambda sweep for CGCNN atom + scaled total-energy loss vs atom-only baseline.

Trains on totals datasets (70/15/15 grouped split) for each domain:
  * baseline: per-atom MAE only (checkpoint by val MAE)
  * lambda runs: atom MAE + lambda * scaled total MSE (checkpoint by val R_tot median)

Default lambdas: 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

Example::

    python train_cgcnn_lambda_sweep.py --skip-build --epochs 300 --plot
    python train_cgcnn_lambda_sweep.py --domain planar --lambda 0.5 --skip-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from train_cgcnn_extensive_comparison import (
    DATASETS,
    DEFAULT_CONFIG,
    SWEEP_JSON,
    SWEEP_LAMBDAS,
    SWEEP_SUMMARY_JSON,
    _train_one,
    build_datasets,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _parse_lambdas(text: str) -> Tuple[float, ...]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No lambda values parsed.")
    return tuple(vals)


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _pick_best_lambda(
    runs: Dict[str, Dict], domain: str, lambdas: Sequence[float]
) -> Optional[Tuple[str, float, Dict]]:
    best_key = None
    best_lambda = None
    best_score = float("inf")
    best_run = None
    for lam in lambdas:
        key = f"{domain}_lambda_{lam:g}"
        run = runs.get(key)
        if not run:
            continue
        score = float(run.get("best_val_r_tot_median", float("inf")))
        if score < best_score:
            best_score = score
            best_key = key
            best_lambda = lam
            best_run = run
    if best_key is None:
        return None
    return best_key, best_lambda, best_run


def _build_summary(runs: Dict[str, Dict], lambdas: Sequence[float]) -> dict:
    summary = {"best_by_domain": {}, "baseline_vs_best": {}}
    for domain in ("point", "planar"):
        baseline_key = f"{domain}_atom_only"
        baseline = runs.get(baseline_key)
        picked = _pick_best_lambda(runs, domain, lambdas)
        if picked:
            best_key, best_lambda, best_run = picked
            summary["best_by_domain"][domain] = {
                "run_key": best_key,
                "lambda_tot": best_lambda,
                "best_val_r_tot_median": best_run["best_val_r_tot_median"],
                "best_test_r_tot_median": best_run["best_test_r_tot_median"],
                "best_val_mae": best_run["best_val_mae"],
                "best_test_mae": best_run["best_test_mae"],
            }
        if baseline and picked:
            best_key, best_lambda, best_run = picked
            summary["baseline_vs_best"][domain] = {
                "baseline_run_key": baseline_key,
                "best_lambda_run_key": best_key,
                "lambda_tot": best_lambda,
                "baseline": {
                    "best_val_mae": baseline["best_val_mae"],
                    "best_test_mae": baseline["best_test_mae"],
                    "best_val_r_tot_median": baseline["best_val_r_tot_median"],
                    "best_test_r_tot_median": baseline["best_test_r_tot_median"],
                },
                "best_lambda": {
                    "best_val_mae": best_run["best_val_mae"],
                    "best_test_mae": best_run["best_test_mae"],
                    "best_val_r_tot_median": best_run["best_val_r_tot_median"],
                    "best_test_r_tot_median": best_run["best_test_r_tot_median"],
                },
                "delta_test_r_tot_median_pct": (
                    best_run["best_test_r_tot_median"]
                    - baseline["best_test_r_tot_median"]
                ),
                "delta_test_mae_eV": (
                    best_run["best_test_mae"] - baseline["best_test_mae"]
                ),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CGCNN lambda sweep for total-energy loss.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--lambdas",
        default=",".join(str(x) for x in SWEEP_LAMBDAS),
        help="Comma-separated lambda_tot values.",
    )
    parser.add_argument(
        "--domain",
        choices=["point", "planar", "both"],
        default="both",
        help="Train on one domain or both.",
    )
    parser.add_argument(
        "--lambda",
        dest="single_lambda",
        type=float,
        default=None,
        help="Run a single lambda (for SLURM array tasks).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip atom-only baseline (use with --lambda for array jobs).",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output-json", default=SWEEP_JSON)
    parser.add_argument("--summary-json", default=SWEEP_SUMMARY_JSON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-prefix",
        default=os.path.join(ROOT, "cgcnn_lambda_sweep"),
    )
    args = parser.parse_args()

    lambdas = (
        (args.single_lambda,)
        if args.single_lambda is not None
        else _parse_lambdas(args.lambdas)
    )

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    domains: Iterable[str]
    if args.domain == "both":
        domains = ("point", "planar")
    else:
        domains = (args.domain,)

    for path in (DATASETS[d] for d in domains):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambdas: {lambdas}")

    payload: dict = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
    runs: Dict[str, Dict] = dict(payload.get("runs", {}))

    for domain in domains:
        dataset_path = DATASETS[domain]
        dataset = torch.load(dataset_path, weights_only=False)
        if not hasattr(dataset[0], "delta_total_eV"):
            raise SystemExit(f"{dataset_path} lacks delta_total_eV; rebuild datasets.")

        baseline_key = f"{domain}_atom_only"
        if not args.skip_baseline and baseline_key not in runs:
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
                legacy_total_loss=True,
            )
            runs[baseline_key]["dataset_path"] = dataset_path
            payload = {
                "metric": args.metric,
                "epochs": args.epochs,
                "seed": args.seed,
                "split": "grouped 70/15/15",
                "lambdas": list(lambdas),
                "config": DEFAULT_CONFIG,
                "runs": runs,
            }
            _save_json(args.output_json, payload)

        for lam in lambdas:
            run_key = f"{domain}_lambda_{lam:g}"
            if run_key in runs:
                print(f"[skip] already trained: {run_key}")
                continue
            print(f"\n>>> Lambda run {run_key}")
            runs[run_key] = _train_one(
                domain=domain,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=DEFAULT_CONFIG,
                use_total_loss=True,
                lambda_tot=lam,
                run_key=run_key,
                checkpoint_metric="r_tot",
                legacy_total_loss=True,
            )
            runs[run_key]["dataset_path"] = dataset_path
            payload = {
                "metric": args.metric,
                "epochs": args.epochs,
                "seed": args.seed,
                "split": "grouped 70/15/15",
                "lambdas": list(lambdas),
                "config": DEFAULT_CONFIG,
                "runs": runs,
            }
            _save_json(args.output_json, payload)

    summary = _build_summary(runs, lambdas)
    payload["summary"] = summary
    _save_json(args.output_json, payload)
    _save_json(args.summary_json, summary)
    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_lambda_sweep.py")
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
