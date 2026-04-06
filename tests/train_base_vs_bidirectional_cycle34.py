"""Train the base GNN (single edge orientation) vs bidirectional edges on cycle34.

Runs both variants for a fixed number of epochs on ``adv_datasets/cycle34_dataset.pt``,
records per-epoch train / val / test metrics, and writes JSON for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import torch

from train_base_vs_cycles import _train_one

DEFAULT_DATASET = os.path.join("adv_datasets", "cycle34_dataset.pt")
DEFAULT_OUTPUT = "base_vs_bidirectional_cycle34.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare undirected-edge storage vs bidirectional message passing "
            "on the cycle34 dataset."
        )
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")

    dataset = torch.load(args.dataset, weights_only=False)

    common: Dict[str, Any] = {
        "dataset": dataset,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_batch_norm": args.use_batch_norm,
        "activation": args.activation,
        "metric": args.metric,
        "seed": args.seed,
    }

    t0 = time.time()
    print("\n" + "=" * 60 + "\n  Model: base (one orientation per pair in data)\n" + "=" * 60)
    curves_base = _train_one(name="base", bidirectional=False, **common)

    print("\n" + "=" * 60 + "\n  Model: bidirectional (two directed edges per pair)\n" + "=" * 60)
    curves_bidir = _train_one(name="bidirectional", bidirectional=True, **common)
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed / 60:.1f} min")

    def _final_test(curves: Dict[str, list]) -> float:
        return float(curves["test"][-1]) if curves["test"] else float("nan")

    output = {
        "dataset_path": args.dataset,
        "epochs": args.epochs,
        "metric": args.metric,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_batch_norm": args.use_batch_norm,
        "activation": args.activation,
        "models": {
            "base": curves_base,
            "bidirectional": curves_bidir,
        },
        "final_test_error": {
            "base": _final_test(curves_base),
            "bidirectional": _final_test(curves_bidir),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
