"""Train CGCNN on residual-ΔPE datasets for point and planar defects.

Mirrors the absolute-target training setup (same hyperparameters, splits, and
metrics) but uses residual targets for both defect types.

1. Optionally builds residual datasets via :mod:`build_residual_datasets`.
2. Trains on ``adv_datasets/cycle34_residual_dataset.pt`` (point).
3. Trains on ``planar_pyg_dataset_residual_c14c15.pt`` (planar C14/C15).
4. Writes ``cgcnn_residual_both_curves.json``.

Example::

    python train_cgcnn_residual_both.py --build-only
    python train_cgcnn_residual_both.py --epochs 300
    python train_cgcnn_residual_both.py --skip-build --epochs 300 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    random_train_val_indices,
    set_seed,
    summarize_split,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS: OrderedDict[str, str] = OrderedDict([
    (
        "defect_residual",
        os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt"),
    ),
    (
        "planar_residual_c14c15",
        os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
    ),
])

CURVES_JSON = os.path.join(ROOT, "cgcnn_residual_both_curves.json")

CHECKPOINT_PATHS = {
    "defect_residual": os.path.join(ROOT, "cgcnn_defect_residual_model.pt"),
    "planar_residual_c14c15": os.path.join(
        ROOT, "cgcnn_planar_residual_c14c15_model.pt"
    ),
}

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


def build_datasets(force: bool = False) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "build_residual_datasets.py")]
    if force:
        cmd.append("--force")
    print("Building residual datasets …")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _train_one(
    name: str,
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    config: Dict,
    val_fraction: Optional[float] = None,
) -> Dict:
    set_seed(seed)

    if val_fraction is not None:
        train_idx, val_idx = random_train_val_indices(
            dataset, seed, val_fraction=val_fraction
        )
        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        test_set = []
        print(
            f"[{name}] delivery fit (random val): train={len(train_set)} "
            f"val={len(val_set)} (val_fraction={val_fraction}, no test)",
            flush=True,
        )
    else:
        train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        test_set = [dataset[i] for i in test_idx]

    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    if test_set:
        summarize_split(f"[{name}] Test", test_set)

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = (
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
        if test_set
        else None
    )

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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []
    best_val = float("inf")
    best_state = None

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

        train_curve.append(metric_value(train_m, metric))
        val_curve.append(metric_value(val_m, metric))
        scheduler.step(val_curve[-1])

        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        test_msg = ""
        if test_loader is not None:
            test_m = evaluate(model, test_loader, device, target_mean, target_std)
            test_curve.append(metric_value(test_m, metric))
            test_msg = f" | test {metric.upper()} {test_curve[-1]:.4f}"

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[{name}] Epoch {epoch:03d} | "
                f"train {metric.upper()} {train_curve[-1]:.4f} | "
                f"val {metric.upper()} {val_curve[-1]:.4f}"
                f"{test_msg} | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_val = evaluate(model, val_loader, device, target_mean, target_std)
    best_test = None
    if test_loader is not None:
        final_test = evaluate(model, test_loader, device, target_mean, target_std)
        best_test = metric_value(final_test, metric)

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "final_train": train_curve[-1],
        "final_val": val_curve[-1],
        "final_test": test_curve[-1] if test_curve else None,
        "best_val": metric_value(final_val, metric),
        "best_test": best_test,
        "target_mode": "residual",
        "best_state": best_state,
        "target_mean": float(target_mean.cpu()),
        "target_std": float(target_std.cpu()),
        "num_parameters": n_params,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "val_fraction": val_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CGCNN on residual-ΔPE point and planar datasets."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--output-json", type=str, default=CURVES_JSON)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After training, write absolute-vs-residual comparison PNG.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=os.path.join(ROOT, "cgcnn_absolute_vs_residual_curves.png"),
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help=(
            "Save best residual CGCNN weights for export "
            "(cgcnn_defect_residual_model.pt / "
            "cgcnn_planar_residual_c14c15_model.pt)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.1), use a train/val-only delivery fit with a "
            "random per-graph val split (no test); checkpoint by best val. "
            "Default keeps the old grouped 70/15/15 split."
        ),
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, Dict] = {}
    for name, path in DATASETS.items():
        if not os.path.isfile(path):
            raise SystemExit(
                f"Dataset not found for {name}: {path}. "
                "Run without --skip-build first."
            )
        print(f"\n=== Training {name} on {path} ===")
        dataset = torch.load(path, weights_only=False)
        curves = _train_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            seed=args.seed,
            config=DEFAULT_CONFIG,
            val_fraction=args.val_fraction,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)

        if args.save_checkpoints:
            ckpt_path = CHECKPOINT_PATHS[name]
            torch.save(
                {
                    "model_state": curves.pop("best_state"),
                    "config": {
                        **DEFAULT_CONFIG,
                        "lr": DEFAULT_CONFIG["lr"],
                        "weight_decay": DEFAULT_CONFIG["weight_decay"],
                        "batch_size": DEFAULT_CONFIG["batch_size"],
                    },
                    "target_mean": curves.pop("target_mean"),
                    "target_std": curves.pop("target_std"),
                    "best_val_mae": curves["best_val"],
                    "test_mae": curves.get("best_test"),
                    "num_parameters": curves.pop("num_parameters"),
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "dataset": path,
                    "target_mode": "residual",
                    "val_fraction": args.val_fraction,
                    "n_train": curves.get("n_train"),
                    "n_val": curves.get("n_val"),
                    "n_test": curves.get("n_test"),
                },
                ckpt_path,
            )
            print(f"[{name}] Saved checkpoint -> {ckpt_path}")
            curves["checkpoint"] = ckpt_path
        else:
            curves.pop("best_state", None)
            curves.pop("target_mean", None)
            curves.pop("target_std", None)
            curves.pop("num_parameters", None)

        results[name] = curves

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "model": "CGCNN",
        "config": DEFAULT_CONFIG,
        "val_fraction": args.val_fraction,
        "datasets": results,
        "notes": (
            "Residual ΔPE targets for both point and planar defects. "
            + (
                f"Delivery fit with val_fraction={args.val_fraction} (no test)."
                if args.val_fraction is not None
                else "Standard 70/15/15 grouped split."
            )
        ),
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved curves to {args.output_json}")

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_absolute_vs_residual.py")
        cmd = [
            sys.executable,
            plot_script,
            "--residual-json",
            args.output_json,
            "--output",
            args.plot_output,
        ]
        print(f"Plotting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
