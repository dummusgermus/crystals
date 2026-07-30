"""Train CGCNN on alternate planar defect-label definitions (legacy absolute PE).

Prefer ``train_cgcnn_planar.py`` for the residual-ΔPE + C14/C15 pipeline that
pairs ``Laves_Planar_Defects`` (initial PE) with ``Laves_Screen`` (relaxed PE).

This script keeps the older absolute-relaxed-PE ablation against label
definitions by relabeling ``planar_pyg_dataset.pt``.  By default:

1. Relabels ``planar_pyg_dataset.pt`` into
   ``planar_pyg_dataset_c14c15.pt`` and ``planar_pyg_dataset_matrix.pt``
   (unless those files already exist).
2. Trains the same CGCNN config used for the labeled-planar baseline.
3. Writes ``cgcnn_planar_new_defs_curves.json`` with keys
   ``planar_c14c15`` and ``planar_matrix``.
4. Optionally merges with existing baseline curve JSONs and plots.

Example::

    python train_cgcnn_planar_new_defs.py --build-only
    python train_cgcnn_planar_new_defs.py --epochs 300 --plot
    python train_cgcnn_planar_new_defs.py --skip-build --epochs 50 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
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

# Alternate planar label definitions (point-defect data is unchanged).
DEFAULT_DEFS: OrderedDict[str, Dict[str, str]] = OrderedDict(
    [
        (
            "planar_c14c15",
            {
                "json": "laves_defect_atoms_c14c15.json",
                "dataset": "planar_pyg_dataset_c14c15.pt",
                "label": "C14/C15 deviation labels",
            },
        ),
        (
            "planar_matrix",
            {
                "json": "laves_defect_atoms_matrix.json",
                "dataset": "planar_pyg_dataset_matrix.pt",
                "label": "matrix-aligned (no defect labels)",
            },
        ),
    ]
)

DEFAULT_OUTPUT_JSON = "cgcnn_planar_new_defs_curves.json"
DEFAULT_BASELINE_JSONS = [
    "cgcnn_defect_vs_planar_curves.json",
    "cgcnn_planar_labeled_curves.json",
]
DEFAULT_PLOT_OUTPUT = "cgcnn_defect_vs_planar_curves.png"

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
            f"lr {current_lr:.1e} | {dt:.1f}s",
            flush=True,
        )

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "final_train": train_curve[-1],
        "final_val": val_curve[-1],
        "final_test": test_curve[-1],
    }


def build_relabeled_datasets(force: bool = False) -> None:
    """Create the C14/C15 and matrix-aligned datasets via relabel_planar_defects."""
    relabel_script = os.path.join(ROOT, "relabel_planar_defects.py")
    for key, meta in DEFAULT_DEFS.items():
        out_path = os.path.join(ROOT, meta["dataset"])
        json_path = os.path.join(ROOT, meta["json"])
        if os.path.isfile(out_path) and not force:
            print(f"[{key}] Dataset exists, skipping build: {out_path}")
            continue
        cmd = [
            sys.executable,
            relabel_script,
            "--defect-atoms-json",
            json_path,
            "--output",
            out_path,
        ]
        if force:
            cmd.append("--force")
        print(f"[{key}] Building {out_path} from {json_path} …")
        subprocess.run(cmd, check=True, cwd=ROOT)


def print_label_summary() -> None:
    """Compare defect-atom counts across baseline and new definitions."""

    def _load(path: str) -> Dict[str, List[int]]:
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            payload = json.load(fh)
        mapping = payload.get("defect_atoms", payload)
        return {
            str(stack): [int(a) for a in ids]
            for stack, ids in mapping.items()
            if not str(stack).startswith("_")
        }

    files = OrderedDict(
        [
            ("baseline", "laves_defect_atoms.json"),
            ("c14c15", "laves_defect_atoms_c14c15.json"),
            ("matrix", "laves_defect_atoms_matrix.json"),
        ]
    )
    maps = {name: _load(path) for name, path in files.items()}
    stacks = sorted(maps["baseline"].keys())
    print("\nDefect-atom counts per stack (definition comparison):")
    header = f"{'stack':<32} {'baseline':>10} {'c14c15':>10} {'matrix':>10}"
    print(header)
    print("-" * len(header))
    for stack in stacks:
        counts = [len(maps[name].get(stack, [])) for name in files]
        print(f"{stack!s:<32} {counts[0]:>10} {counts[1]:>10} {counts[2]:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CGCNN on new planar defect-label definitions and write "
            "curves for comparison against existing baselines."
        )
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only (re)build relabeled datasets; do not train.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild datasets even if missing (fail if absent).",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild relabeled datasets even if outputs already exist.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After training, merge with baseline JSONs and write the comparison PNG.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=DEFAULT_PLOT_OUTPUT,
        help="PNG path used when --plot is set.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        choices=list(DEFAULT_DEFS.keys()),
        help="Train only these definition keys (default: all).",
    )
    args = parser.parse_args()

    print_label_summary()

    if not args.skip_build:
        build_relabeled_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    keys = args.only or list(DEFAULT_DEFS.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    results: Dict[str, Dict] = {}
    for key in keys:
        meta = DEFAULT_DEFS[key]
        path = os.path.join(ROOT, meta["dataset"])
        if not os.path.isfile(path):
            raise SystemExit(
                f"Dataset not found for {key}: {path}. "
                "Run without --skip-build first."
            )
        print(f"\n=== Training {key} ({meta['label']}) on {path} ===")
        dataset = torch.load(path, weights_only=False)
        curves = _train_one(
            name=key,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            seed=args.seed,
            config=DEFAULT_CONFIG,
        )
        curves["dataset_path"] = meta["dataset"]
        curves["defect_atoms_json"] = meta["json"]
        curves["label"] = meta["label"]
        curves["num_graphs"] = len(dataset)
        results[key] = curves

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "model": "CGCNN",
        "config": DEFAULT_CONFIG,
        "datasets": results,
        "notes": (
            "New planar defect-label definitions only; point-defect baseline "
            "unchanged. Compare against defect / planar / planar_labeled curves."
        ),
    }
    out_json = os.path.join(ROOT, args.output_json)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved curves to {out_json}")

    print("\nFinal test metrics:")
    for key, curves in results.items():
        print(
            f"  {key}: test {args.metric.upper()} = {curves['final_test']:.4f} "
            f"({DEFAULT_DEFS[key]['label']})"
        )

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_defect_vs_planar.py")
        inputs = [
            os.path.join(ROOT, p)
            for p in DEFAULT_BASELINE_JSONS
            if os.path.isfile(os.path.join(ROOT, p))
        ]
        inputs.append(out_json)
        cmd = [
            sys.executable,
            plot_script,
            "--input",
            *inputs,
            "--output",
            os.path.join(ROOT, args.plot_output),
        ]
        print(f"\nPlotting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
