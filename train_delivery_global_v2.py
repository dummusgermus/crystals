"""Train four delivery models (CGCNN + Transformer × point/planar) with global-loss v2.

Production settings (k13 point, full-cell CGCNN loss, sweep-optimal lambdas).
Same 90/10 within-group split as predictions_new (seed 42).

Example::

    python train_delivery_global_v2.py --skip-build --epochs 2000
    python train_delivery_global_v2.py --model point_cgcnn --epochs 300 --force-split
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import torch

from build_delivery_datasets import datasets_ready
from delivery_global_v2 import (
    ALL_MODEL_KEYS,
    BENCHMARK_JSON,
    CHECKPOINTS,
    CURVES_JSON,
    DELIVERY_SEED,
    DELIVERY_VAL_FRACTION,
    DELIVERY_VERSION,
    LAMBDA_BY_MODEL,
    SPLIT_JSON_DEFAULT,
    TOTALS_DATASETS,
    benchmark_delivery_inference,
    build_or_update_split_json,
    save_delivery_split,
    train_delivery_global_v2,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _parse_models(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(ALL_MODEL_KEYS)
    keys = [p.strip() for p in text.split(",") if p.strip()]
    for key in keys:
        if key not in CHECKPOINTS:
            raise ValueError(f"Unknown model {key!r}; choose from {ALL_MODEL_KEYS}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train delivery global-v2 models for predictions_new."
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--model",
        default="all",
        help="all or comma-separated: point_cgcnn, planar_cgcnn, ...",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Recompute and overwrite delivery_split_indices.json",
    )
    parser.add_argument("--split-json", default=SPLIT_JSON_DEFAULT)
    parser.add_argument("--output-json", default=CURVES_JSON)
    parser.add_argument("--benchmark-json", default=BENCHMARK_JSON)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip val-set inference timing after training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models whose checkpoint file already exists.",
    )
    args = parser.parse_args()

    should_build = not args.skip_build and (args.force_build or not datasets_ready())
    if should_build:
        cmd = [sys.executable, os.path.join(ROOT, "build_delivery_datasets.py")]
        if args.force_build:
            cmd.append("--force")
        subprocess.run(cmd, check=True, cwd=ROOT)
    elif not args.skip_build:
        print("Delivery datasets already present; skipping build.", flush=True)
    if args.build_only:
        print("Build-only done.")
        return

    models = _parse_models(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Delivery version: {DELIVERY_VERSION}")
    print(f"Split seed={DELIVERY_SEED} val_fraction={DELIVERY_VAL_FRACTION}")
    print(f"Lambdas: {LAMBDA_BY_MODEL}")
    print(f"Totals datasets: {TOTALS_DATASETS}")

    split_payload = build_or_update_split_json(
        path=args.split_json, force=args.force_split
    )
    save_delivery_split(split_payload, args.split_json)
    print(f"Split indices -> {args.split_json}")

    results: Dict[str, Dict] = {}
    benchmarks: Dict[str, Dict] = {}
    if os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
        results = dict(payload.get("runs", {}))
    if os.path.isfile(args.benchmark_json):
        with open(args.benchmark_json, encoding="utf-8") as fh:
            benchmarks = dict(json.load(fh).get("models", {}))

    for model_key in models:
        ckpt_path = CHECKPOINTS[model_key]
        if args.resume and os.path.isfile(ckpt_path):
            print(f"[skip train] checkpoint exists: {model_key}")
        else:
            domain, kind = model_key.split("_", 1)
            results[model_key] = train_delivery_global_v2(
                domain=domain,
                model_kind=kind,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                checkpoint_path=ckpt_path,
                split_payload=split_payload,
            )

        if not args.skip_benchmark and os.path.isfile(ckpt_path):
            print(f"[benchmark] {model_key} …", flush=True)
            benchmarks[model_key] = benchmark_delivery_inference(
                model_key=model_key,
                checkpoint_path=ckpt_path,
                split_payload=split_payload,
                device=device,
            )
            print(
                f"[benchmark] {model_key}: "
                f"{benchmarks[model_key]['inference_ms_per_graph']:.2f} ms/graph "
                f"({benchmarks[model_key]['val_graphs']} val graphs)",
                flush=True,
            )

        payload = {
            "version": DELIVERY_VERSION,
            "epochs": args.epochs,
            "seed": DELIVERY_SEED,
            "val_fraction": DELIVERY_VAL_FRACTION,
            "split_json": args.split_json,
            "lambda_by_model": LAMBDA_BY_MODEL,
            "totals_datasets": TOTALS_DATASETS,
            "checkpoint_metric": "val_r_tot_median",
            "runs": results,
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        with open(args.benchmark_json, "w", encoding="utf-8") as fh:
            json.dump(
                {"version": DELIVERY_VERSION, "models": benchmarks},
                fh,
                indent=2,
            )

    print(f"\nSaved training curves -> {args.output_json}")
    print(f"Saved inference benchmarks -> {args.benchmark_json}")
    for key in models:
        if key not in results:
            print(f"  {key}: (skipped — no training result this run)")
            continue
        r = results[key]
        extra_key = (
            "final_val_r_tot_full_median"
            if r.get("total_target_mode") == "graph"
            else "final_val_r_tot_graph_median"
        )
        extra = r.get(extra_key, float("nan"))
        print(
            f"  {key}: target={r.get('total_target_mode')} "
            f"val R_tot={r['best_val_r_tot_median']:.1f}% "
            f"alt R_tot={extra:.1f}% "
            f"val MAE={r['best_val_mae']:.4f} -> {r['checkpoint']}"
        )


if __name__ == "__main__":
    main()
