"""Train CGCNN on the labeled planar dataset and log epoch curves to JSON.

Uses ``planar_pyg_dataset.pt`` (full Laves_Screen graphs with planar-fault
``is_defect`` / ``dist_to_defect`` labels).  Writes curves to a separate JSON
file so existing comparison data is not overwritten.

Example::

    python train_cgcnn_planar_labeled_curves.py
    python train_cgcnn_planar_labeled_curves.py --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

DEFAULT_DATASET = "planar_pyg_dataset.pt"
DEFAULT_OUTPUT_JSON = "cgcnn_planar_labeled_curves.json"
DATASET_KEY = "planar_labeled"

DEFAULT_CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
    bidirectional=True,
    lr=2e-3,
    weight_decay=0.0,
    batch_size=8,
)


def train_planar_curves(
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    config: Dict,
) -> Dict[str, List[float]]:
    """Train CGCNN on one dataset; return per-epoch train / val / test curves."""
    set_seed(seed)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split(f"[{DATASET_KEY}] Train", train_set)
    summarize_split(f"[{DATASET_KEY}] Val", val_set)
    summarize_split(f"[{DATASET_KEY}] Test", test_set)

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
        bidirectional=config["bidirectional"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            if metric == "mae":
                loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                loss = per_graph_mse_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device, target_mean, target_std)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        test_m = evaluate(model, test_loader, device, target_mean, target_std)

        train_curve.append(metric_value(train_m, metric))
        val_curve.append(metric_value(val_m, metric))
        test_curve.append(metric_value(test_m, metric))

        scheduler.step(val_curve[-1])

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{DATASET_KEY}] Epoch {epoch:03d} | "
            f"train {metric.upper()} {train_curve[-1]:.4f} | "
            f"val {metric.upper()} {val_curve[-1]:.4f} | "
            f"test {metric.upper()} {test_curve[-1]:.4f} | "
            f"lr {current_lr:.1e} | {dt:.1f}s"
        )

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "final_train": train_curve[-1],
        "final_val": val_curve[-1],
        "final_test": test_curve[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CGCNN on the labeled planar dataset and write epoch curves "
            "to a new JSON file."
        )
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    dataset = torch.load(args.dataset, weights_only=False)
    curves = train_planar_curves(
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        metric=args.metric,
        seed=args.seed,
        config=DEFAULT_CONFIG,
    )
    curves["dataset_path"] = args.dataset
    curves["num_graphs"] = len(dataset)

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "model": "CGCNN",
        "config": DEFAULT_CONFIG,
        "datasets": {DATASET_KEY: curves},
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved curves to {args.output_json}")


if __name__ == "__main__":
    main()
