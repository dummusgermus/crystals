"""Train four delivery models (CGCNN + Transformer × point/planar) with global-loss v2.

Same 90/10 within-group split as original predictions/ delivery (seed 42).
Checkpoints by validation R_tot median. Saves split indices to JSON.

Example::

    python train_delivery_global_v2.py --skip-build --epochs 2000
    python train_delivery_global_v2.py --model point_cgcnn --epochs 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import torch

from delivery_global_v2 import (
    CHECKPOINTS,
    DELIVERY_SEED,
    DELIVERY_VAL_FRACTION,
    LAMBDA_BY_DOMAIN,
    SPLIT_JSON_DEFAULT,
    build_or_update_split_json,
    save_delivery_split,
    train_delivery_global_v2,
)
from train_cgcnn_extensive_comparison import build_datasets

ROOT = os.path.dirname(os.path.abspath(__file__))

ALL_MODELS = (
    "point_cgcnn",
    "planar_cgcnn",
    "point_transformer",
    "planar_transformer",
)

CURVES_JSON = os.path.join(ROOT, "delivery_global_v2_curves.json")


def _parse_models(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(ALL_MODELS)
    keys = [p.strip() for p in text.split(",") if p.strip()]
    for key in keys:
        if key not in CHECKPOINTS:
            raise ValueError(f"Unknown model {key!r}; choose from {ALL_MODELS}")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models whose checkpoint file already exists.",
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    models = _parse_models(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Delivery split seed={DELIVERY_SEED} val_fraction={DELIVERY_VAL_FRACTION}")
    print(f"Lambdas: {LAMBDA_BY_DOMAIN}")

    split_payload = build_or_update_split_json(
        path=args.split_json, force=args.force_split
    )
    save_delivery_split(split_payload, args.split_json)
    print(f"Split indices -> {args.split_json}")

    results: Dict[str, Dict] = {}
    if os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
        results = dict(payload.get("runs", {}))

    for model_key in models:
        ckpt_path = CHECKPOINTS[model_key]
        if args.resume and os.path.isfile(ckpt_path):
            print(f"[skip] checkpoint exists: {model_key}")
            continue
        domain, kind = model_key.split("_", 1)
        ckpt_path = CHECKPOINTS[model_key]
        results[model_key] = train_delivery_global_v2(
            domain=domain,
            model_kind=kind,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            checkpoint_path=ckpt_path,
            split_payload=split_payload,
        )
        payload = {
            "epochs": args.epochs,
            "seed": DELIVERY_SEED,
            "val_fraction": DELIVERY_VAL_FRACTION,
            "split_json": args.split_json,
            "lambda_by_domain": LAMBDA_BY_DOMAIN,
            "loss": "global_v2",
            "checkpoint_metric": "val_r_tot_median",
            "runs": results,
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    print(f"\nSaved training curves -> {args.output_json}")
    for key in models:
        if key not in results:
            print(f"  {key}: (skipped — no result in this run)")
            continue
        r = results[key]
        print(
            f"  {key}: val R_tot={r['best_val_r_tot_median']:.1f}% "
            f"val MAE={r['best_val_mae']:.4f} -> {r['checkpoint']}"
        )


if __name__ == "__main__":
    main()
