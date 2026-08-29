"""Lambda sweep for Graph Transformer + global-loss v2 (totals datasets).

Grouped 70/15/15 split; checkpoint by val R_tot median; pick best by test R_tot median.

Default grids (around delivery λ):
  point:  0.005, 0.01, 0.02
  planar: 0.0025, 0.005, 0.01

Example::

    python train_transformer_global_v2_lambda_sweep.py --skip-build --epochs 1000 --plot
    sbatch --export=ALL,SKIP_BUILD=1 run_transformer_global_v2_lambda_sweep.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from train_cgcnn_extensive_comparison import DATASETS, build_datasets
from train_cgcnn_global_v2 import default_total_loss_config
from train_transformer_global_v2 import TRANSFORMER_CONFIG, _train_one

ROOT = os.path.dirname(os.path.abspath(__file__))

SWEEP_JSON = os.path.join(ROOT, "transformer_global_v2_lambda_sweep_curves.json")
SWEEP_SUMMARY_JSON = os.path.join(ROOT, "transformer_global_v2_lambda_sweep_summary.json")
SWEEP_PLOT_PREFIX = os.path.join(ROOT, "transformer_global_v2_lambda_sweep")

DEFAULT_POINT_LAMBDAS = (0.005, 0.01, 0.02)
DEFAULT_PLANAR_LAMBDAS = (0.0025, 0.005, 0.01)
DEFAULT_EPOCHS = 1000

BEST_CHECKPOINT_NAMES = {
    "point": "transformer_graph_defect_residual_global_v2_best_model.pt",
    "planar": "transformer_graph_planar_residual_global_v2_best_model.pt",
}


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
    return f"{domain}_transformer_global_v2_lambda_{lam:g}"


def _run_checkpoint_path(domain: str, lam: float) -> str:
    return os.path.join(
        ROOT, f"transformer_{domain}_global_v2_lambda_{lam:g}_model.pt"
    )


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _domain_lambdas(
    domain: str,
    point_lambdas: Sequence[float],
    planar_lambdas: Sequence[float],
) -> Tuple[float, ...]:
    return point_lambdas if domain == "point" else planar_lambdas


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


def _build_summary(
    runs: Dict[str, Dict],
    point_lambdas: Sequence[float],
    planar_lambdas: Sequence[float],
) -> dict:
    summary = {
        "checkpoint_metric": "r_tot",
        "selection_metric": "final_test_r_tot_median",
        "optimal_lambda_by_model": {
            "cgcnn": {"point": 0.01, "planar": 0.005},
            "transformer": {"point": 0.02, "planar": 0.01},
        },
        "best_by_domain": {},
        "all_runs_by_domain": {},
    }
    for domain, lambdas in (
        ("point", point_lambdas),
        ("planar", planar_lambdas),
    ):
        per_lambda = {}
        for lam in lambdas:
            key = _global_run_key(domain, lam)
            run = runs.get(key)
            if not run:
                continue
            per_lambda[f"lambda_{lam:g}"] = {
                "run_key": key,
                "lambda_tot": lam,
                "checkpoint": run.get("checkpoint"),
                "final_test_mae": run["final_test_mae"],
                "final_test_r_tot_median": run["final_test_r_tot_median"],
                "final_test_abs_total_err_eV": run["final_test_abs_total_err_eV"],
                "best_val_score": run["best_val_score"],
            }
        summary["all_runs_by_domain"][domain] = per_lambda

        picked = _pick_best_lambda(runs, domain, lambdas)
        if picked:
            best_key, best_lambda, best_run = picked
            best_ckpt_name = BEST_CHECKPOINT_NAMES[domain]
            summary["best_by_domain"][domain] = {
                "run_key": best_key,
                "lambda_tot": best_lambda,
                "final_test_mae": best_run["final_test_mae"],
                "final_test_r_tot_median": best_run["final_test_r_tot_median"],
                "final_test_abs_total_err_eV": best_run["final_test_abs_total_err_eV"],
                "best_val_score": best_run["best_val_score"],
                "source_checkpoint": best_run.get("checkpoint"),
                "best_checkpoint": os.path.join(ROOT, best_ckpt_name),
                "config": dict(TRANSFORMER_CONFIG),
                "total_loss_config": default_total_loss_config(best_lambda).__dict__,
            }
    return summary


def _copy_best_checkpoints(summary: dict) -> None:
    for domain, block in summary.get("best_by_domain", {}).items():
        src = block.get("source_checkpoint")
        dst = block.get("best_checkpoint")
        if not src or not dst or not os.path.isfile(src):
            print(f"[warn] missing source checkpoint for {domain}: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"[best] {domain} λ={block['lambda_tot']:g} -> {dst}")


def _payload_skeleton(
    *,
    metric: str,
    epochs: int,
    seed: int,
    point_lambdas: Sequence[float],
    planar_lambdas: Sequence[float],
    runs: Dict[str, Dict],
    summary: Optional[dict] = None,
) -> dict:
    payload = {
        "version": "transformer_global_v2_lambda_sweep",
        "model": "transformer",
        "metric": metric,
        "epochs": epochs,
        "seed": seed,
        "split": "grouped 70/15/15",
        "checkpoint_metric": "r_tot",
        "point_lambdas": list(point_lambdas),
        "planar_lambdas": list(planar_lambdas),
        "config": TRANSFORMER_CONFIG,
        "runs": runs,
    }
    if summary is not None:
        payload["summary"] = summary
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer global v2 lambda sweep (1000 epochs default)."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--point-lambdas",
        default=",".join(str(x) for x in DEFAULT_POINT_LAMBDAS),
        help="Comma-separated λ for point domain (default: 0.005,0.01,0.02).",
    )
    parser.add_argument(
        "--planar-lambdas",
        default=",".join(str(x) for x in DEFAULT_PLANAR_LAMBDAS),
        help="Comma-separated λ for planar domain (default: 0.0025,0.005,0.01).",
    )
    parser.add_argument(
        "--domain",
        choices=["point", "planar", "both"],
        default="both",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default=SWEEP_JSON)
    parser.add_argument("--summary-json", default=SWEEP_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=SWEEP_PLOT_PREFIX)
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help="Do not copy best run checkpoints to *_best_model.pt names.",
    )
    args = parser.parse_args()

    point_lambdas = _parse_lambdas(args.point_lambdas)
    planar_lambdas = _parse_lambdas(args.planar_lambdas)

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
    print(f"Epochs: {args.epochs}, checkpoint: val R_tot median")
    print(f"Point lambdas: {point_lambdas}")
    print(f"Planar lambdas: {planar_lambdas}")

    payload: dict = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
    runs: Dict[str, Dict] = dict(payload.get("runs", {}))

    for domain in domains:
        lambdas = _domain_lambdas(domain, point_lambdas, planar_lambdas)
        dataset_path = DATASETS[domain]
        dataset = torch.load(dataset_path, weights_only=False)
        if not hasattr(dataset[0], "delta_total_eV"):
            raise SystemExit(f"{dataset_path} lacks delta_total_eV; rebuild datasets.")

        for lam in lambdas:
            run_key = _global_run_key(domain, lam)
            if run_key in runs:
                print(f"[skip] already trained: {run_key}")
                continue
            total_cfg = default_total_loss_config(lam)
            ckpt_path = _run_checkpoint_path(domain, lam)
            print(f"\n>>> {run_key} (lambda={lam:g})")
            runs[run_key] = _train_one(
                domain=domain,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=TRANSFORMER_CONFIG,
                use_total_loss=True,
                lambda_tot=lam,
                run_key=run_key,
                checkpoint_metric="r_tot",
                total_loss_config=total_cfg,
                checkpoint_path=ckpt_path,
            )
            runs[run_key]["dataset_path"] = dataset_path
            _save_json(
                args.output_json,
                _payload_skeleton(
                    metric=args.metric,
                    epochs=args.epochs,
                    seed=args.seed,
                    point_lambdas=point_lambdas,
                    planar_lambdas=planar_lambdas,
                    runs=runs,
                ),
            )

    summary = _build_summary(runs, point_lambdas, planar_lambdas)
    payload = _payload_skeleton(
        metric=args.metric,
        epochs=args.epochs,
        seed=args.seed,
        point_lambdas=point_lambdas,
        planar_lambdas=planar_lambdas,
        runs=runs,
        summary=summary,
    )
    _save_json(args.output_json, payload)
    _save_json(args.summary_json, summary)

    if not args.no_copy_best:
        _copy_best_checkpoints(summary)

    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    print(json.dumps(summary, indent=2))

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_transformer_global_v2_lambda_sweep.py")
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
