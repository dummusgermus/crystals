"""Train CGCNN on planar graphs with C14/C15 deviation labels.

Same ML task as the point-defect pipeline (:mod:`graph_maker`):

* Node feature ``per_atom_pe`` = initial (unrelaxed) pe/atom
* Target ``y`` = absolute relaxed pe/atom

Only the input structures differ (planar Laves stacks vs bulk point defects).

Example::

    python train_cgcnn_planar.py --build-only
    python train_cgcnn_planar.py --epochs 300 --plot
    python train_cgcnn_planar.py --skip-build --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(ROOT, "planar_pyg_dataset_c14c15.pt")
STATS_PATH = os.path.join(ROOT, "planar_pyg_dataset_c14c15_stats.json")
MODEL_PATH = os.path.join(ROOT, "cgcnn_planar_c14c15_model.pt")
CURVES_JSON = os.path.join(ROOT, "cgcnn_planar_c14c15_curves.json")
DEFECT_ATOMS_JSON = os.path.join(ROOT, "laves_defect_atoms_c14c15.json")
INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")

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


def build_dataset(force: bool = False) -> None:
    """Build the C14/C15 planar dataset (absolute relaxed PE target)."""
    if os.path.isfile(DATASET_PATH) and not force:
        print(f"Dataset exists, skipping build: {DATASET_PATH}")
        return

    cmd = [
        sys.executable,
        os.path.join(ROOT, "planar_graph_maker.py"),
        "--initial-simulations-dir",
        INITIAL_DIR,
        "--relaxed-simulations-dir",
        RELAXED_DIR,
        "--defect-atoms-json",
        DEFECT_ATOMS_JSON,
        "--target-mode",
        "absolute",
        "--output",
        DATASET_PATH,
        "--stats-output",
        STATS_PATH,
    ]
    print("Building planar C14/C15 dataset (absolute relaxed PE target) …")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def _train(
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    config: Dict,
) -> Dict:
    set_seed(seed)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split("[planar_c14c15] Train", train_set)
    summarize_split("[planar_c14c15] Val", val_set)
    summarize_split("[planar_c14c15] Test", test_set)

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

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

    t_total = time.time()
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

        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        dt = time.time() - t0
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | "
                f"train {metric.upper()} {train_curve[-1]:.4f} | "
                f"val {metric.upper()} {val_curve[-1]:.4f} | "
                f"test {metric.upper()} {test_curve[-1]:.4f} | "
                f"lr {lr:.1e} | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_val = evaluate(model, val_loader, device, target_mean, target_std)
    final_test = evaluate(model, test_loader, device, target_mean, target_std)

    print(f"\nTraining complete in {time.time() - t_total:.0f}s")
    print(f"Best val {metric.upper()}: {metric_value(final_val, metric):.4f}")
    print(f"Test {metric.upper()} (at best val): {metric_value(final_test, metric):.4f}")

    checkpoint = {
        "model_state": best_state,
        "config": dict(config),
        "target_mean": float(target_mean.cpu()),
        "target_std": float(target_std.cpu()),
        "best_val_mae": final_val.mae,
        "test_mae": final_test.mae,
        "num_parameters": n_params,
        "epochs": epochs,
        "seed": seed,
        "dataset": DATASET_PATH,
        "target_mode": "absolute",
        "defect_atoms_json": DEFECT_ATOMS_JSON,
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "final_train": train_curve[-1],
        "final_val": val_curve[-1],
        "final_test": test_curve[-1],
        "best_val": metric_value(final_val, metric),
        "best_test": metric_value(final_test, metric),
        "dataset_path": os.path.basename(DATASET_PATH),
        "defect_atoms_json": os.path.basename(DEFECT_ATOMS_JSON),
        "label": "planar defect data (C14/C15)",
        "num_graphs": len(dataset),
        "target_mode": "absolute",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CGCNN on planar data with absolute relaxed PE target "
            "(same task as point-defect graph_maker)."
        )
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only (re)build the dataset; do not train.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild even if the dataset is missing (fail if absent).",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild the dataset even if the output already exists.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=CURVES_JSON,
        help="Where to write train/val/test curves.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After training, plot point-defect vs planar curves.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=os.path.join(ROOT, "cgcnn_defect_vs_planar_curves.png"),
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_dataset(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    if not os.path.isfile(DATASET_PATH):
        raise SystemExit(
            f"Dataset not found: {DATASET_PATH}. "
            "Run without --skip-build first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Dataset: {DATASET_PATH}")

    dataset = torch.load(DATASET_PATH, weights_only=False)
    curves = _train(
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        metric=args.metric,
        seed=args.seed,
        config=DEFAULT_CONFIG,
    )

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "model": "CGCNN",
        "config": DEFAULT_CONFIG,
        "datasets": {"planar_c14c15": curves},
        "notes": (
            "Planar absolute relaxed PE with C14/C15 labels — same target "
            "definition as point-defect graph_maker. Initial PE from "
            "Laves_Planar_Defects; relaxed PE from Laves_Screen."
        ),
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved curves to {args.output_json}")

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_defect_vs_planar.py")
        cmd = [
            sys.executable,
            plot_script,
            "--input",
            os.path.join(ROOT, "cgcnn_defect_vs_planar_curves.json"),
            args.output_json,
            "--only",
            "defect",
            "planar_c14c15",
            "--output",
            args.plot_output,
        ]
        print(f"\nPlotting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
