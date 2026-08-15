"""Long local run: Graph Transformer on point residual (1200 epochs).

Uses the current default hyperparameters. Saves per-epoch curves so you can
see when / whether training stabilizes.

Example::

    python train_transformer_point_residual_1200.py
"""

from __future__ import annotations

import json
import os

import torch

from train_transformer_absolute_residual import (
    DATASETS,
    GRAPH_CONFIG,
    ROOT,
    _train_one,
)

EPOCHS = 1200
SCORE_LAST_N = 50
OUTPUT_JSON = os.path.join(ROOT, "transformer_point_residual_1200_curves.json")
NAME = "defect_residual"


def main() -> None:
    meta = DATASETS[NAME]
    path = meta["path"]
    if not os.path.isfile(path):
        raise SystemExit(f"Dataset not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = dict(GRAPH_CONFIG)
    print(f"Device: {device}")
    print(f"Dataset: {path}")
    print(f"Epochs: {EPOCHS}")
    print(f"Config: {config}")

    dataset = torch.load(path, weights_only=False)
    result = _train_one(
        name=NAME,
        dataset=dataset,
        device=device,
        epochs=EPOCHS,
        metric="mae",
        seed=42,
        architecture="graph",
        config=config,
        target_mode=meta["target_mode"],
        checkpoint_path=None,
        score_last_n=SCORE_LAST_N,
        quiet_every=10,
    )
    result["dataset_path"] = path
    result["num_graphs"] = len(dataset)

    payload = {
        "metric": "mae",
        "epochs": EPOCHS,
        "seed": 42,
        "score_last_n": SCORE_LAST_N,
        "model": "GraphTransformer",
        "architecture": "graph",
        "config": config,
        "datasets": {NAME: result},
        "notes": (
            "Long local stability run on point residual with current defaults. "
            f"Primary score = mean of last {SCORE_LAST_N} epoch MAEs."
        ),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved curves to {OUTPUT_JSON}")
    print(
        f"last-{SCORE_LAST_N} val={result['last_n_val_mean']:.6f}"
        f"±{result['last_n_val_std']:.6f}  "
        f"test={result['last_n_test_mean']:.6f}"
        f"±{result['last_n_test_std']:.6f}"
    )


if __name__ == "__main__":
    main()
