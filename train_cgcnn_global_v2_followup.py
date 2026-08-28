"""300-epoch follow-up: global v2 at lambda 0.01 and 0.005 vs atom-only baseline.

Global runs checkpoint by val R_tot median; baseline checkpoints by val MAE.

Example::

    python train_cgcnn_global_v2_followup.py --skip-build --epochs 300 --plot
    python train_cgcnn_global_v2_followup.py --lambdas 0.01 --skip-baseline --resume
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
    _train_one,
    build_datasets,
)
from train_cgcnn_global_v2 import default_total_loss_config

ROOT = os.path.dirname(os.path.abspath(__file__))

FOLLOWUP_JSON = os.path.join(ROOT, "cgcnn_global_v2_followup_curves.json")
FOLLOWUP_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_global_v2_followup_summary.json")
FOLLOWUP_PLOT_PREFIX = os.path.join(ROOT, "cgcnn_global_v2_followup")

DEFAULT_LAMBDAS = (0.01, 0.005)
DEFAULT_EPOCHS = 300


def _parse_lambdas(text: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("No lambda values parsed.")
    return tuple(vals)


def _global_run_key(domain: str, lam: float) -> str:
    return f"{domain}_global_v2_lambda_{lam:g}"


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
        key = _global_run_key(domain, lam)
        run = runs.get(key)
        if not run:
            continue
        score = float(run.get("final_test_r_tot_median", float("inf")))
        if score < best_score:
            best_score = score
            best_key = key
            best_lambda = lam
            best_run = run
    if best_key is None:
        return None
    return best_key, best_lambda, best_run


def _build_summary(runs: Dict[str, Dict], lambdas: Sequence[float]) -> dict:
    summary = {
        "baseline_checkpoint_metric": "mae",
        "global_checkpoint_metric": "r_tot",
        "best_by_domain": {},
        "baseline_vs_lambdas": {},
    }
    for domain in ("point", "planar"):
        baseline_key = f"{domain}_atom_only"
        baseline = runs.get(baseline_key)
        picked = _pick_best_lambda(runs, domain, lambdas)
        if picked:
            best_key, best_lambda, best_run = picked
            summary["best_by_domain"][domain] = {
                "run_key": best_key,
                "lambda_tot": best_lambda,
                "final_test_mae": best_run["final_test_mae"],
                "final_test_r_tot_median": best_run["final_test_r_tot_median"],
                "final_test_abs_total_err_eV": best_run["final_test_abs_total_err_eV"],
            }

        per_lambda = {}
        if baseline:
            for lam in lambdas:
                key = _global_run_key(domain, lam)
                run = runs.get(key)
                if not run:
                    continue
                per_lambda[f"lambda_{lam:g}"] = {
                    "run_key": key,
                    "lambda_tot": lam,
                    "final_test_mae": run["final_test_mae"],
                    "final_test_r_tot_median": run["final_test_r_tot_median"],
                    "final_test_abs_total_err_eV": run["final_test_abs_total_err_eV"],
                    "delta_test_r_tot_median_pct": (
                        run["final_test_r_tot_median"]
                        - baseline["final_test_r_tot_median"]
                    ),
                    "delta_test_mae_eV": (
                        run["final_test_mae"] - baseline["final_test_mae"]
                    ),
                    "delta_test_abs_total_err_eV": (
                        run["final_test_abs_total_err_eV"]
                        - baseline["final_test_abs_total_err_eV"]
                    ),
                }
        if baseline or per_lambda:
            summary["baseline_vs_lambdas"][domain] = {
                "baseline_run_key": baseline_key,
                "baseline": {
                    "final_test_mae": baseline["final_test_mae"],
                    "final_test_r_tot_median": baseline["final_test_r_tot_median"],
                    "final_test_abs_total_err_eV": baseline["final_test_abs_total_err_eV"],
                }
                if baseline
                else None,
                "lambdas": per_lambda,
            }
    return summary


def _payload_skeleton(
    *,
    metric: str,
    epochs: int,
    seed: int,
    lambdas: Sequence[float],
    runs: Dict[str, Dict],
    summary: Optional[dict] = None,
) -> dict:
    payload = {
        "version": "global_v2_followup",
        "metric": metric,
        "epochs": epochs,
        "seed": seed,
        "split": "grouped 70/15/15",
        "lambdas": list(lambdas),
        "baseline_checkpoint_metric": "mae",
        "global_checkpoint_metric": "r_tot",
        "total_loss_config_template": default_total_loss_config(lambdas[0]).__dict__,
        "config": DEFAULT_CONFIG,
        "runs": runs,
    }
    if summary is not None:
        payload["summary"] = summary
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global v2 follow-up: baseline + lambda 0.01/0.005 at 300 epochs."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--lambdas",
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="Comma-separated lambda values (default: 0.01,0.005).",
    )
    parser.add_argument(
        "--domain",
        choices=["point", "planar", "both"],
        default="both",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default=FOLLOWUP_JSON)
    parser.add_argument("--summary-json", default=FOLLOWUP_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-prefix",
        default=FOLLOWUP_PLOT_PREFIX,
    )
    args = parser.parse_args()

    lambdas = _parse_lambdas(args.lambdas)

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

    for domain in domains:
        if not os.path.isfile(DATASETS[domain]):
            raise SystemExit(f"Missing dataset: {DATASETS[domain]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambdas: {lambdas}, epochs: {args.epochs}")
    print("Baseline checkpoint: val MAE | Global checkpoint: val R_tot median")

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
            )
            runs[baseline_key]["dataset_path"] = dataset_path
            _save_json(
                args.output_json,
                _payload_skeleton(
                    metric=args.metric,
                    epochs=args.epochs,
                    seed=args.seed,
                    lambdas=lambdas,
                    runs=runs,
                ),
            )

        for lam in lambdas:
            run_key = _global_run_key(domain, lam)
            if run_key in runs:
                print(f"[skip] already trained: {run_key}")
                continue
            total_cfg = default_total_loss_config(lam)
            print(f"\n>>> Global v2 {run_key} (lambda={lam:g})")
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
                total_loss_config=total_cfg,
                legacy_total_loss=False,
            )
            runs[run_key]["dataset_path"] = dataset_path
            _save_json(
                args.output_json,
                _payload_skeleton(
                    metric=args.metric,
                    epochs=args.epochs,
                    seed=args.seed,
                    lambdas=lambdas,
                    runs=runs,
                ),
            )

    summary = _build_summary(runs, lambdas)
    payload = _payload_skeleton(
        metric=args.metric,
        epochs=args.epochs,
        seed=args.seed,
        lambdas=lambdas,
        runs=runs,
        summary=summary,
    )
    _save_json(args.output_json, payload)
    _save_json(args.summary_json, summary)
    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    print(json.dumps(summary, indent=2))

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_global_v2_followup.py")
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
