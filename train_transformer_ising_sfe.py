"""Train Graph Transformer on the 20261020 Ising/SFE total-energy dataset.

Same protocol as ``train_cgcnn_ising_sfe.py`` (material-grouped 70/15/15,
graph-level ``total_energy_eV``), but with
:class:`gnn_models.GraphTransformerGraphRegressor`.

If ``adv_datasets/ising_sfe_split_indices.json`` already exists (e.g. from the
CGCNN run) and ``--seed`` matches, that split is reused for a fair comparison.

Example::

    python train_transformer_ising_sfe.py --epochs 300 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_graph_transformer_graph_model_from_dataset
from train_cgcnn_ising_sfe import (
    DEFAULT_DATASET,
    DEFAULT_SEED,
    MIN_ENERGY_EV,
    _material_key,
    _plot_results,
    _sample_weights,
    energy_loss,
    evaluate,
    graph_target,
    material_grouped_split_indices,
    sanitize_dataset_for_batching,
    summarize_split,
)
from train_single import set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))

CURVES_JSON = os.path.join(ROOT, "transformer_ising_sfe_curves.json")
SUMMARY_JSON = os.path.join(ROOT, "transformer_ising_sfe_summary.json")
SPLIT_JSON = os.path.join(ROOT, "adv_datasets", "ising_sfe_split_indices.json")
CKPT_PATH = os.path.join(ROOT, "transformer_ising_sfe_total_energy_model.pt")
PLOT_PREFIX = os.path.join(ROOT, "transformer_ising_sfe")

# Defaults aligned with GRAPH_CONFIG in train_transformer_absolute_residual.py
DEFAULT_CONFIG = dict(
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    aggregation="mean",
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=8,  # dense attention; graphs up to ~384 atoms
    huber_delta=1.0,
    scale_eps=1.0,
    outlier_max_abs_eV=5000.0,
    gradient_norm=1.0,
)


def _load_or_make_split(dataset, seed: int, split_json: str):
    """Reuse CGCNN split when present and seed matches; else create one."""
    if os.path.isfile(split_json):
        with open(split_json, encoding="utf-8") as fh:
            payload = json.load(fh)
        if int(payload.get("seed", -1)) == int(seed) and all(
            k in payload for k in ("train_idx", "val_idx", "test_idx")
        ):
            train_idx = np.array(payload["train_idx"], dtype=int)
            val_idx = np.array(payload["val_idx"], dtype=int)
            test_idx = np.array(payload["test_idx"], dtype=int)
            assign = payload.get(
                "materials",
                {"train": [], "val": [], "test": []},
            )
            print(f"Reusing existing split -> {split_json}", flush=True)
            return train_idx, val_idx, test_idx, assign

    train_idx, val_idx, test_idx, assign = material_grouped_split_indices(
        dataset, seed
    )
    os.makedirs(os.path.dirname(os.path.abspath(split_json)) or ".", exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": seed,
                "split": "material-grouped 70/15/15",
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_materials_train": len(assign["train"]),
                "n_materials_val": len(assign["val"]),
                "n_materials_test": len(assign["test"]),
                "materials": assign,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
            },
            fh,
            indent=2,
        )
    print(f"Saved split -> {split_json}", flush=True)
    return train_idx, val_idx, test_idx, assign


def train(
    *,
    dataset,
    device: torch.device,
    epochs: int,
    seed: int,
    config: Dict,
    split_json: str,
    ckpt_path: str,
) -> Dict:
    set_seed(seed)
    train_idx, val_idx, test_idx, assign = _load_or_make_split(
        dataset, seed, split_json
    )

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    print(
        f"\n=== ising_sfe | GraphTransformer | total_energy_eV | "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} ===",
        flush=True,
    )
    summarize_split("[ising_sfe] Train", train_set)
    summarize_split("[ising_sfe] Val", val_set)
    summarize_split("[ising_sfe] Test", test_set)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.stack([graph_target(d) for d in train_set]).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)
    print(
        f"[ising_sfe] target norm mean={float(target_mean):.4f} "
        f"std={float(target_std):.4f}",
        flush=True,
    )

    model = build_graph_transformer_graph_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation=config["activation"],
        aggregation=config["aggregation"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ising_sfe] params={n_params:,} "
        f"in_dim={dataset[0].x.size(-1)} edge_dim={dataset[0].edge_attr.size(-1)}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=16, min_lr=1e-6
    )

    curves = {
        "train_mae_eV": [],
        "val_mae_eV": [],
        "test_mae_eV": [],
        "train_rel_median": [],
        "val_rel_median": [],
        "test_rel_median": [],
        "train_abs_err_eV": [],
        "val_abs_err_eV": [],
        "test_abs_err_eV": [],
    }
    best_val_rel = float("inf")
    best_state = None
    grad_clip = float(config.get("gradient_norm", 0.0) or 0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch).view(-1) * target_std + target_mean
            true = batch.y.view(-1)
            weights = _sample_weights(true, config["outlier_max_abs_eV"])
            loss = energy_loss(
                pred,
                true,
                scale_eps=config["scale_eps"],
                huber_delta=config["huber_delta"],
                weights=weights,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        train_m = evaluate(model, train_loader, device, target_mean, target_std)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        test_m = evaluate(model, test_loader, device, target_mean, target_std)

        curves["train_mae_eV"].append(train_m.mae_eV)
        curves["val_mae_eV"].append(val_m.mae_eV)
        curves["test_mae_eV"].append(test_m.mae_eV)
        curves["train_rel_median"].append(train_m.median_rel_pct)
        curves["val_rel_median"].append(val_m.median_rel_pct)
        curves["test_rel_median"].append(test_m.median_rel_pct)
        curves["train_abs_err_eV"].append(train_m.mean_abs_err_eV)
        curves["val_abs_err_eV"].append(val_m.mean_abs_err_eV)
        curves["test_abs_err_eV"].append(test_m.mean_abs_err_eV)

        score = val_m.median_rel_pct if val_m.n_rel > 0 else val_m.mae_eV
        if score < best_val_rel:
            best_val_rel = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_m.mae_eV)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[ising_sfe-GT] ep {epoch:04d} | val MAE {val_m.mae_eV:.4f} eV | "
                f"val rel med {val_m.median_rel_pct:.2f}% | "
                f"test rel med {test_m.median_rel_pct:.2f}% | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = evaluate(model, train_loader, device, target_mean, target_std)
    final_val = evaluate(model, val_loader, device, target_mean, target_std)
    final_test = evaluate(model, test_loader, device, target_mean, target_std)

    torch.save(
        {
            "model_state": best_state,
            "architecture": "graph_transformer_graph_total",
            "config": config,
            "target_mean": float(target_mean.cpu()),
            "target_std": float(target_std.cpu()),
            "domain": "ising_sfe",
            "target": "total_energy_eV",
            "best_val_rel_median_pct": best_val_rel,
            "epochs": epochs,
            "seed": seed,
            "split": "material-grouped 70/15/15",
            "num_parameters": n_params,
            "in_dim": int(dataset[0].x.size(-1)),
            "edge_dim": int(dataset[0].edge_attr.size(-1)),
        },
        ckpt_path,
    )
    print(f"[ising_sfe-GT] Saved checkpoint -> {ckpt_path}", flush=True)

    test_preds: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for data in test_set:
            data_dev = data.to(device)
            pred = float(
                (model(data_dev).view(-1) * target_std + target_mean).cpu().item()
            )
            true = float(data.y.view(-1).cpu().item())
            err = pred - true
            rel = (
                100.0 * abs(err) / abs(true)
                if abs(true) >= MIN_ENERGY_EV
                else float("nan")
            )
            test_preds.append(
                {
                    "material": _material_key(data),
                    "kind": getattr(data, "kind", None),
                    "phase": getattr(data, "phase", None),
                    "fault": getattr(data, "fault", None),
                    "polytype": getattr(data, "polytype", None),
                    "true_eV": true,
                    "pred_eV": pred,
                    "err_eV": err,
                    "abs_err_eV": abs(err),
                    "rel_pct": rel,
                }
            )

    return {
        "domain": "ising_sfe",
        "checkpoint": ckpt_path,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_materials_train": len(assign.get("train", [])),
        "n_materials_val": len(assign.get("val", [])),
        "n_materials_test": len(assign.get("test", [])),
        "best_val_rel_median_pct": best_val_rel,
        "final_train_mae_eV": final_train.mae_eV,
        "final_val_mae_eV": final_val.mae_eV,
        "final_test_mae_eV": final_test.mae_eV,
        "final_train_rel_median": final_train.median_rel_pct,
        "final_val_rel_median": final_val.median_rel_pct,
        "final_test_rel_median": final_test.median_rel_pct,
        "final_test_abs_err_median_eV": final_test.median_abs_err_eV,
        "test_predictions": test_preds,
        **curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Graph Transformer graph regressor on Ising/SFE total-energy graphs."
        )
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--output-json", default=CURVES_JSON)
    parser.add_argument("--summary-json", default=SUMMARY_JSON)
    parser.add_argument("--split-json", default=SPLIT_JSON)
    parser.add_argument("--checkpoint", default=CKPT_PATH)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=PLOT_PREFIX)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(
            f"Missing dataset: {args.dataset}\n"
            "Place ising_sfe_cycle34_total_energy_dataset.pt under adv_datasets/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {os.path.abspath(args.dataset)}")
    print(f"Split: material-grouped 70/15/15 seed={args.seed}")
    print("Model: GraphTransformerGraphRegressor")

    dataset = torch.load(args.dataset, weights_only=False)
    print(f"Loaded {len(dataset)} graphs", flush=True)
    sanitize_dataset_for_batching(dataset)
    print("Sanitized optional attrs for PyG batching", flush=True)

    config = dict(DEFAULT_CONFIG)
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr

    run = train(
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
        config=config,
        split_json=args.split_json,
        ckpt_path=args.checkpoint,
    )

    payload = {
        "epochs": args.epochs,
        "seed": args.seed,
        "dataset": os.path.abspath(args.dataset),
        "split": "material-grouped 70/15/15",
        "target": "total_energy_eV",
        "checkpoint_metric": "val_rel_median_pct",
        "model": "GraphTransformerGraphRegressor",
        "config": config,
        "runs": {"ising_sfe": run},
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    summary = {
        "target": "total_energy_eV (graph-level)",
        "model": "GraphTransformerGraphRegressor (mean-pool scalar output)",
        "split": "material-grouped 70/15/15",
        "epochs": args.epochs,
        "seed": args.seed,
        "dataset": os.path.abspath(args.dataset),
        "final_test_mae_eV": run["final_test_mae_eV"],
        "final_test_rel_median_pct": run["final_test_rel_median"],
        "final_test_abs_err_median_eV": run["final_test_abs_err_median_eV"],
        "best_val_rel_median_pct": run["best_val_rel_median_pct"],
        "checkpoint": run["checkpoint"],
        "n_train": run["n_train"],
        "n_val": run["n_val"],
        "n_test": run["n_test"],
        "n_materials_train": run["n_materials_train"],
        "n_materials_val": run["n_materials_val"],
        "n_materials_test": run["n_materials_test"],
    }
    with open(args.summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    print(
        f"  test rel med={run['final_test_rel_median']:.2f}% "
        f"test MAE={run['final_test_mae_eV']:.4f} eV "
        f"test |err| med={run['final_test_abs_err_median_eV']:.4f} eV"
    )

    if args.plot:
        _plot_results(
            payload,
            args.plot_prefix,
            title="Ising/SFE graph-level total energy Graph Transformer",
        )


if __name__ == "__main__":
    main()
