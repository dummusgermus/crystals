"""Train the CGCNN GatedConv model with optimal parameters and save checkpoint.

Uses cycle34_dataset.pt with bidirectional edges, trains for 300 epochs, and
saves the trained model as cgcnn_model.pt.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

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
OUTPUT_PATH = "cgcnn_model.pt"
EPOCHS = 300
SEED = 42

CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
)
LR = 2e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 8


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Epochs: {EPOCHS}")

    dataset = torch.load(DATASET_PATH, weights_only=False)

    set_seed(SEED)
    train_idx, val_idx, test_idx = grouped_split_indices(dataset, SEED)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split("Train", train_set)
    summarize_split("Val", val_set)
    summarize_split("Test", test_set)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    set_seed(SEED)
    model = build_gated_model_from_dataset(
        dataset, bidirectional=True, **CONFIG
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    best_val_mae = float("inf")
    best_state = None

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

        scheduler.step(val_m.mae)
        dt = time.time() - t0

        if val_m.mae < best_val_mae:
            best_val_mae = val_m.mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | train MAE {train_m.mae:.4f} | "
                f"val MAE {val_m.mae:.4f} | test MAE {test_m.mae:.4f} | "
                f"lr {lr:.1e} | {dt:.1f}s"
            )

    total_time = time.time() - t_total

    # Reload best weights and evaluate
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_test = evaluate(model, test_loader, device, target_mean, target_std)
    final_val = evaluate(model, val_loader, device, target_mean, target_std)

    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best val MAE: {final_val.mae:.4f}")
    print(f"Test MAE (at best val): {final_test.mae:.4f}")

    checkpoint: Dict = {
        "model_state": best_state,
        "config": {
            **CONFIG,
            "bidirectional": True,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
        },
        "target_mean": float(target_mean.cpu()),
        "target_std": float(target_std.cpu()),
        "best_val_mae": final_val.mae,
        "test_mae": final_test.mae,
        "num_parameters": n_params,
        "epochs": EPOCHS,
        "seed": SEED,
        "dataset": DATASET_PATH,
    }
    torch.save(checkpoint, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
