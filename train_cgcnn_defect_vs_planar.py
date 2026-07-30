"""Train the same CGCNN on full and unit-box scaled defect datasets; log curves.

Runs one training job per dataset with identical hyperparameters, records
per-epoch train / val / test MAE, and writes a JSON file for comparison
plotting.  Does not save model checkpoints.

Compares:
  - adv_datasets/cycle34_dataset.pt          (physical coordinates)
  - adv_datasets_scaled/scaled_cycle34_dataset.pt  (unit-box scaled cut-outs)

Example::

    python train_cgcnn_defect_vs_planar.py
    python train_cgcnn_defect_vs_planar.py --epochs 50 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
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

DEFAULT_DATASETS: OrderedDict[str, str] = OrderedDict([
    ("defect", os.path.join("adv_datasets", "cycle34_dataset.pt")),
    ("scaled", os.path.join("adv_datasets_scaled", "scaled_cycle34_dataset.pt")),
])

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

PLOT_LABELS = {
    "defect": "point defect data",
    "scaled": "scaled point defect data",
}
PLOT_COLORS = {"defect": "C0", "scaled": "C1"}
PLOT_TITLE = "CGCNN - point defect vs. scaled point defect data"


def _train_one(
    name: str,
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

    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    summarize_split(f"[{name}] Test", test_set)

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
            f"[{name}] Epoch {epoch:03d} | "
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


def _plot_curves(payload: Dict, output_path: str, curve: str) -> None:
    metric = str(payload.get("metric", "mae")).upper()
    datasets: Dict[str, Dict] = payload["datasets"]

    if curve == "all":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
        for ax, curve_name in zip(axes, ("train", "val", "test")):
            for key in datasets:
                series = datasets[key].get(curve_name, [])
                if not series:
                    continue
                ax.plot(
                    range(1, len(series) + 1),
                    series,
                    label=PLOT_LABELS.get(key, key),
                    color=PLOT_COLORS.get(key),
                )
            ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle(PLOT_TITLE, fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in datasets:
            series = datasets[key].get(curve, [])
            if not series:
                continue
            ax.plot(
                range(1, len(series) + 1),
                series,
                label=PLOT_LABELS.get(key, key),
                color=PLOT_COLORS.get(key),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{curve.capitalize()} {metric}")
        fig.suptitle(PLOT_TITLE, fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the same CGCNN on full and scaled point-defect datasets; "
            "write comparison curves to JSON."
        )
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument(
        "--output-json",
        type=str,
        default="cgcnn_defect_vs_scaled_curves.json",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a comparison PNG after training.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="cgcnn_defect_vs_scaled_curves.png",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which curve(s) to plot when --plot is set.",
    )
    parser.add_argument(
        "--defect-dataset",
        type=str,
        default=DEFAULT_DATASETS["defect"],
    )
    parser.add_argument(
        "--scaled-dataset",
        type=str,
        default=DEFAULT_DATASETS["scaled"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_paths = OrderedDict([
        ("defect", args.defect_dataset),
        ("scaled", args.scaled_dataset),
    ])

    results: Dict[str, Dict] = {}
    for name, path in dataset_paths.items():
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping '{name}'.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Training CGCNN on: {name}  ({path})")
        print(f"{'=' * 60}")

        dataset = torch.load(path, weights_only=False)
        curves = _train_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            seed=args.seed,
            config=DEFAULT_CONFIG,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)
        results[name] = curves

    if not results:
        raise SystemExit("No datasets were trained — check dataset paths.")

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "model": "CGCNN",
        "config": DEFAULT_CONFIG,
        "datasets": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved curves to {args.output_json}")

    if args.plot:
        _plot_curves(payload, args.plot_output, args.curve)


if __name__ == "__main__":
    main()
