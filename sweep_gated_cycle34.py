"""Exhaustive hyperparameter sweep for bidirectional GatedConv (CGCNN) on cycle34.

Trains every combination in the grid for 300 epochs and saves the mean
test/val MAE over the last 10 epochs.  Results are written incrementally
so partial runs can be resumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

DEFAULT_DATASET = os.path.join("adv_datasets", "cycle34_dataset.pt")
DEFAULT_OUTPUT = "sweep_gated_cycle34_v3.json"

SWEEP_SPACE: Dict[str, list] = {
    "hidden_dim": [128, 256],
    "num_layers": [2, 3, 4],
    "dropout": [0.0, 0.05, 0.1],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [0.0, 1e-4],
    "batch_size": [8, 16, 32],
}

FIXED_ACTIVATION = "silu"
FIXED_BATCH_NORM = False
LAST_N = 10


def _generate_configs() -> List[Dict[str, Any]]:
    """Generate the full grid of configs (exhaustive, no sampling)."""
    keys = list(SWEEP_SPACE.keys())
    all_combos = list(itertools.product(*(SWEEP_SPACE[k] for k in keys)))
    return [dict(zip(keys, combo)) for combo in all_combos]


def _train_config(
    cfg: Dict[str, Any],
    dataset: list,
    train_idx,
    val_idx,
    test_idx,
    device: torch.device,
    epochs: int,
    train_seed: int,
) -> Dict[str, Any]:
    """Train one configuration and return summary metrics."""

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    bs = cfg["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    set_seed(train_seed)
    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_batch_norm=FIXED_BATCH_NORM,
        activation=FIXED_ACTIVATION,
        bidirectional=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    val_maes: List[float] = []
    test_maes: List[float] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()

        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        scheduler.step(val_m.mae)

        if epoch > epochs - LAST_N:
            test_m = evaluate(model, test_loader, device, target_mean, target_std)
            val_maes.append(val_m.mae)
            test_maes.append(test_m.mae)

    total_time = time.time() - t0

    return {
        "config": cfg,
        "num_parameters": n_params,
        "total_train_time": round(total_time, 1),
        "val_mae_last10": val_maes,
        "test_mae_last10": test_maes,
        "mean_val_mae": round(sum(val_maes) / len(val_maes), 6),
        "mean_test_mae": round(sum(test_maes) / len(test_maes), 6),
    }


def _save(path: str, meta: Dict, results: List[Dict]) -> None:
    payload = {**meta, "results": results}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GatedConv hyperparameter sweep on cycle34.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Fixed: act={FIXED_ACTIVATION}, bn={FIXED_BATCH_NORM}")

    dataset = torch.load(args.dataset, weights_only=False)

    set_seed(args.seed)
    train_idx, val_idx, test_idx = grouped_split_indices(dataset, args.seed)
    summarize_split("Train", [dataset[i] for i in train_idx])
    summarize_split("Val", [dataset[i] for i in val_idx])
    summarize_split("Test", [dataset[i] for i in test_idx])

    configs = _generate_configs()
    print(f"Exhaustive grid: {len(configs)} configs")

    meta = {
        "dataset_path": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_configs": len(configs),
        "sweep_space": {k: [str(x) for x in v] for k, v in SWEEP_SPACE.items()},
        "fixed": {
            "activation": FIXED_ACTIVATION,
            "batch_norm": FIXED_BATCH_NORM,
        },
        "last_n_epochs": LAST_N,
    }

    # Resume: load existing results and skip completed configs
    completed: Dict[str, Dict] = {}
    if os.path.isfile(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            key = json.dumps(r["config"], sort_keys=True)
            completed[key] = r
        print(f"Resuming: {len(completed)} configs already completed")

    results: List[Dict] = list(completed.values())

    for i, cfg in enumerate(configs):
        key = json.dumps(cfg, sort_keys=True)
        if key in completed:
            continue

        print(f"\n{'='*60}")
        print(f"  Config {i+1}/{len(configs)}: {cfg}")
        print(f"{'='*60}")

        try:
            result = _train_config(
                cfg, dataset, train_idx, val_idx, test_idx,
                device, args.epochs, args.seed,
            )
            results.append(result)
            completed[key] = result

            print(
                f"  -> val MAE: {result['mean_val_mae']:.4f} | "
                f"test MAE: {result['mean_test_mae']:.4f} | "
                f"time: {result['total_train_time']:.0f}s | "
                f"params: {result['num_parameters']:,}"
            )
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({"config": cfg, "error": str(e)})

        _save(args.output, meta, results)

    # Final summary: top 10 by val MAE
    valid = [r for r in results if "mean_val_mae" in r]
    valid.sort(key=lambda r: r["mean_val_mae"])
    print(f"\n{'='*60}")
    print(f"  Sweep complete: {len(valid)}/{len(configs)} succeeded")
    print(f"  Top 10 by validation MAE:")
    print(f"{'='*60}")
    for rank, r in enumerate(valid[:10], 1):
        c = r["config"]
        print(
            f"  #{rank:2d}  val={r['mean_val_mae']:.4f}  test={r['mean_test_mae']:.4f}  "
            f"hd={c['hidden_dim']}  nl={c['num_layers']}  "
            f"dr={c['dropout']}  lr={c['lr']}  wd={c['weight_decay']}  "
            f"bs={c['batch_size']}  "
            f"params={r['num_parameters']:,}  time={r['total_train_time']:.0f}s"
        )

    _save(args.output, meta, results)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
