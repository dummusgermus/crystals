"""Training run for the ct-UAE-fronted gated GNN (winning configuration).

Mirrors :mod:`train_single.py` (same split, loss, metrics, LR schedule)
but replaces the per-atom ``type`` scalar with a pretrained, **frozen**
8-D ct-UAE embedding from :mod:`ct_uae_pretrain`.  This is the only UAE
configuration that matched the scalar-type baseline on the binary-alloy
PE dataset; wider embeddings and trainable embeddings were measured to
degrade performance by 15-80%.  See ``uae_gated_result.json`` for the
per-epoch curves of the reference run.

The script writes only a JSON file with per-epoch train/val/test curves:
no model checkpoints, no plots.

Expected dataset format
-----------------------
Each ``Data`` in the input ``.pt`` must carry a ``z`` long tensor
(``0`` = vacancy, otherwise atomic number) plus the usual ``x,
edge_index, edge_attr, y``.  Regenerate with the current
``adv_graph_maker.py`` if missing.

Typical use
-----------
Pretrain the 8-D UAE once::

    python ct_uae_pretrain.py

Then train the UAE-fronted gated GNN::

    python train_uae_single.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_uae_gated_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

# Winning configuration (see repository README / uae_gated_result.json).
UAE_EMB_DIM = 8
UAE_VOCAB_SIZE = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the UAE-fronted gated GNN with the winning configuration: "
            "pretrained 8-D frozen UAE + bidirectional gated CGCNN."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join("adv_datasets_uae", "cycle34_dataset.pt"),
        help=(
            "Cycle-augmented PyG dataset with per-node Data.z "
            "(built by adv_graph_maker.py)."
        ),
    )
    parser.add_argument(
        "--uae-ckpt",
        type=str,
        default=os.path.join("uae_embeddings", f"uae_emb{UAE_EMB_DIM}.pt"),
        help="Pretrained UAE checkpoint produced by ct_uae_pretrain.py.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="uae_gated_result.json",
        help="Output JSON with per-epoch train/val/test curves.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.dataset, weights_only=False)
    if not hasattr(dataset[0], "z") or dataset[0].z is None:
        raise SystemExit(
            f"Dataset at {args.dataset!r} has no per-node atomic numbers "
            "(Data.z). Rebuild it with adv_graph_maker.py."
        )
    if not os.path.isfile(args.uae_ckpt):
        raise SystemExit(
            f"UAE checkpoint not found: {args.uae_ckpt!r}. "
            "Pretrain it first with: python ct_uae_pretrain.py"
        )

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, args.seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split("Train", train_set)
    summarize_split("Val", val_set)
    summarize_split("Test", test_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_uae_gated_model_from_dataset(
        dataset,
        uae_ckpt_path=args.uae_ckpt,
        uae_emb_dim=UAE_EMB_DIM,
        uae_vocab_size=UAE_VOCAB_SIZE,
        freeze_uae=True,
        drop_type_scalar=True,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        bidirectional=True,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Model parameters: {n_trainable:,} trainable / {n_total:,} total "
        f"(UAE frozen at emb_dim={UAE_EMB_DIM})"
    )

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_test = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            if args.metric == "mae":
                loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                loss = per_graph_mse_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader, device, target_mean, target_std)
        val_metrics = evaluate(model, val_loader, device, target_mean, target_std)
        test_metrics = evaluate(model, test_loader, device, target_mean, target_std)

        val_value = metric_value(val_metrics, args.metric)
        test_value = metric_value(test_metrics, args.metric)
        train_curve.append(metric_value(train_metrics, args.metric))
        val_curve.append(val_value)
        test_curve.append(test_value)

        if val_value < best_val:
            best_val = val_value
            best_epoch = epoch
            best_test = test_value

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[uae-gated] Epoch {epoch:03d} | "
            f"train {args.metric.upper()} {train_curve[-1]:.4f} | "
            f"val {args.metric.upper()} {val_value:.4f} | "
            f"test {args.metric.upper()} {test_value:.4f} | "
            f"lr {current_lr:.1e} | time {dt:.1f}s"
        )
        scheduler.step(val_value)

    print(
        f"Best val {args.metric.upper()} = {best_val:.4f} at epoch {best_epoch} "
        f"(test @ best-val = {best_test:.4f})"
    )

    output: Dict[str, Any] = {
        "tag": "uae-gated-bidir",
        "dataset_path": args.dataset,
        "uae_ckpt_path": args.uae_ckpt,
        "uae_emb_dim": UAE_EMB_DIM,
        "uae_vocab_size": UAE_VOCAB_SIZE,
        "uae_frozen": True,
        "dropped_type_scalar": True,
        "bidirectional": True,
        "inner": "gated",
        "epochs": args.epochs,
        "metric": args.metric,
        "seed": args.seed,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "activation": args.activation,
        },
        "num_parameters_trainable": int(n_trainable),
        "num_parameters_total": int(n_total),
        "curves": {
            "train": train_curve,
            "val": val_curve,
            "test": test_curve,
        },
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "test_at_best_val": float(best_test),
        "final_train": float(train_curve[-1]) if train_curve else float("nan"),
        "final_val": float(val_curve[-1]) if val_curve else float("nan"),
        "final_test": float(test_curve[-1]) if test_curve else float("nan"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved per-epoch curves to {args.output}")


if __name__ == "__main__":
    main()
