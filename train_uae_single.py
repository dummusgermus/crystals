"""Single training run for a ct-UAE-fronted node PE regressor.

Mirrors :mod:`train_single.py` (same split logic, loss, metrics, LR
schedule, plotting) but swaps the per-atom ``type`` scalar for a learned
ct-UAE-style atom embedding produced by :mod:`ct_uae_pretrain`.

Expected dataset format
-----------------------
Each ``Data`` object in the input ``.pt`` must carry a ``z`` long tensor
(``0`` for vacancy, otherwise atomic number) plus the usual ``x,
edge_index, edge_attr, y``.  If you're loading an older
``pyg_dataset.pt`` that predates the UAE update, regenerate it with the
current ``graph_maker.py``::

    python graph_maker.py

Typical use
-----------
First pretrain the UAE once (see :mod:`ct_uae_pretrain`)::

    python ct_uae_pretrain.py

Then train a UAE-fronted gated GNN::

    python train_uae_single.py --uae-ckpt uae_embeddings/uae_emb128.pt \
        --inner gated --hidden-dim 256 --num-layers 2 --epochs 200

Or train a freshly-initialised UAE alongside the main task (no
pretraining, UAE layer learned end-to-end)::

    python train_uae_single.py --no-pretrained
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import (
    build_uae_base_model_from_dataset,
    build_uae_gated_model_from_dataset,
    build_uae_tower_model_from_dataset,
)
from train_single import (
    Metrics,
    compute_metrics,
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)


def _build_model(args, dataset):
    kwargs = dict(
        uae_ckpt_path=None if args.no_pretrained else args.uae_ckpt,
        uae_emb_dim=args.uae_emb_dim,
        uae_vocab_size=args.uae_vocab_size,
        freeze_uae=args.freeze_uae,
        drop_type_scalar=not args.keep_type_scalar,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
        bidirectional=args.bidirectional,
    )
    if args.inner == "gated":
        return build_uae_gated_model_from_dataset(dataset, **kwargs)
    if args.inner == "tower":
        return build_uae_tower_model_from_dataset(
            dataset, num_towers=args.num_towers, **kwargs
        )
    if args.inner == "base":
        return build_uae_base_model_from_dataset(dataset, **kwargs)
    raise ValueError(f"Unknown --inner: {args.inner}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single training run for a UAE-fronted node PE regressor."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pyg_dataset_uae.pt",
        help=(
            "PyG dataset with per-node Data.z (produced by the updated "
            "graph_maker.py). Defaults to pyg_dataset_uae.pt so the original "
            "pyg_dataset.pt is left untouched."
        ),
    )
    # UAE encoder
    parser.add_argument(
        "--uae-ckpt",
        type=str,
        default="uae_embeddings/uae_emb128.pt",
        help="Path to a UAE checkpoint from ct_uae_pretrain.py.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialise the UAE layer from scratch instead of loading --uae-ckpt.",
    )
    parser.add_argument("--uae-emb-dim", type=int, default=128)
    parser.add_argument("--uae-vocab-size", type=int, default=100)
    parser.add_argument("--freeze-uae", action="store_true")
    parser.add_argument(
        "--keep-type-scalar",
        action="store_true",
        help="Concatenate the UAE embedding *in addition to* the raw type scalar.",
    )
    # GNN backbone
    parser.add_argument(
        "--inner",
        choices=["gated", "tower", "base"],
        default="gated",
        help="Inner node-regressor architecture to wrap in the UAE encoder.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-towers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--bidirectional", action="store_true")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--metric", type=str, default="mae",
                        choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=str, default="train_uae_curve.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.dataset, weights_only=False)
    if not hasattr(dataset[0], "z") or dataset[0].z is None:
        raise SystemExit(
            f"Dataset at {args.dataset!r} has no per-node atomic numbers (Data.z). "
            "Regenerate it with the current graph_maker.py:  python graph_maker.py"
        )

    if not args.no_pretrained:
        if not os.path.isfile(args.uae_ckpt):
            raise SystemExit(
                f"UAE checkpoint not found: {args.uae_ckpt!r}. "
                "Run ct_uae_pretrain.py first, or pass --no-pretrained."
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

    model = _build_model(args, dataset).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_trainable/1e6:.2f}M trainable / "
          f"{n_total/1e6:.2f}M total (UAE frozen={args.freeze_uae})")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    test_curve: List[float] = []
    val_curve: List[float] = []
    train_curve: List[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_test = float("inf")

    tag = f"uae-{args.inner}" + ("-bidir" if args.bidirectional else "")
    print(f"Training tag: {tag}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
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
            epoch_loss += loss.item() * batch.num_graphs
        epoch_loss /= max(len(train_loader.dataset), 1)

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
            f"[{tag}] Epoch {epoch:03d} | "
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

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label=f"train {args.metric.upper()}")
    plt.plot(val_curve, label=f"val {args.metric.upper()}")
    plt.plot(test_curve, label=f"test {args.metric.upper()}")
    plt.axvline(best_epoch - 1, color="grey", linestyle=":", label=f"best val (ep {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel(args.metric.upper())
    plt.title(f"{tag}: {args.metric.upper()} over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
