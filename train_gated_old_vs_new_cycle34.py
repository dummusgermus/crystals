"""Compare CGCNN GatedConv with old vs new (sweep-optimised) hyperparameters.

Trains both on cycle34_dataset.pt with bidirectional edges for 300 epochs,
saves JSON results and generates a line plot.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    per_graph_mae_loss,
    set_seed,
    summarize_split,
)

DATASET_PATH = os.path.join("adv_datasets", "cycle34_dataset.pt")
OUTPUT_JSON = "gated_old_vs_new_cycle34.json"
OUTPUT_PLOT = "gated_old_vs_new_cycle34_plot.png"
EPOCHS = 300
SEED = 42

CONFIGS = {
    "old (hd=256, bs=16, wd=0)": dict(
        hidden_dim=256, num_layers=2, dropout=0.0,
        use_batch_norm=False, activation="silu",
        lr=1e-3, weight_decay=0.0, batch_size=16,
    ),
    "new (hd=128, bs=32, wd=1e-5)": dict(
        hidden_dim=128, num_layers=2, dropout=0.0,
        use_batch_norm=False, activation="silu",
        lr=1e-3, weight_decay=1e-5, batch_size=32,
    ),
}


def _train_one(
    name: str,
    cfg: Dict[str, Any],
    dataset,
    train_idx,
    val_idx,
    test_idx,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Dict[str, Any]:
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    bs = cfg["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    set_seed(SEED)
    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg["use_batch_norm"],
        activation=cfg["activation"],
        bidirectional=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []
    epoch_times: List[float] = []

    t_total = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device, target_mean, target_std)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        test_m = evaluate(model, test_loader, device, target_mean, target_std)

        train_curve.append(train_m.mae)
        val_curve.append(val_m.mae)
        test_curve.append(test_m.mae)
        dt = time.time() - t0
        epoch_times.append(dt)

        scheduler.step(val_m.mae)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[{name}] Epoch {epoch:3d} | "
                f"train {train_m.mae:.4f} | val {val_m.mae:.4f} | "
                f"test {test_m.mae:.4f} | {dt:.1f}s"
            )

    total_time = time.time() - t_total
    return {
        "config": cfg,
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "epoch_times": epoch_times,
        "total_train_time": total_time,
        "num_parameters": n_params,
    }


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset = torch.load(DATASET_PATH, weights_only=False)
    set_seed(SEED)
    train_idx, val_idx, test_idx = grouped_split_indices(dataset, SEED)
    summarize_split("Train", [dataset[i] for i in train_idx])
    summarize_split("Val", [dataset[i] for i in val_idx])
    summarize_split("Test", [dataset[i] for i in test_idx])

    train_targets = torch.cat([dataset[i].y for i in train_idx], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    all_results: Dict[str, Dict] = {}
    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        all_results[name] = _train_one(
            name, cfg, dataset, train_idx, val_idx, test_idx,
            device, target_mean, target_std,
        )

    # Save JSON
    output = {
        "dataset": DATASET_PATH,
        "epochs": EPOCHS,
        "seed": SEED,
        "models": all_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {OUTPUT_JSON}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = {"old (hd=256, bs=16, wd=0)": "C0", "new (hd=128, bs=32, wd=1e-5)": "C1"}
    for name, res in all_results.items():
        xs = range(1, len(res["test"]) + 1)
        ax.plot(xs, res["test"], label=name, color=colors[name], linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MAE")
    ax.set_ylim(0.01, 0.02)
    ax.set_title("CGCNN GatedConv: old vs sweep-optimised hyperparameters")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    footer_parts = []
    for name, res in all_results.items():
        final = res["test"][-1]
        footer_parts.append(
            f"{name}:  test MAE = {final:.4f}  |  "
            f"time = {res['total_train_time']:.0f}s  |  "
            f"params = {res['num_parameters']:,}"
        )
    fig.text(0.5, 0.01, "\n".join(footer_parts),
             ha="center", va="bottom", fontsize=8, family="monospace")
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
