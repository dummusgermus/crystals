"""Train CGCNN graph regressor on the 20261020 Ising/SFE total-energy dataset.

Uses the same gated graph-level architecture as ``train_cgcnn_graph_total.py``,
adapted for absolute cell total energy (no per-atom PE / residual):

* Dataset: ``adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt``
* Target: ``y`` = total potential energy (eV), shape ``[1, 1]``
* Split: **material-grouped** 70/15/15 (no binary leaks across splits)
* Checkpoint: best validation median relative error (%)

Example::

    python train_cgcnn_ising_sfe.py --epochs 300 --plot
    python train_cgcnn_ising_sfe.py --dataset adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_graph_model_from_dataset
from train_single import set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET = os.path.join(
    ROOT, "adv_datasets", "ising_sfe_cycle34_total_energy_dataset.pt"
)
CURVES_JSON = os.path.join(ROOT, "cgcnn_ising_sfe_curves.json")
SUMMARY_JSON = os.path.join(ROOT, "cgcnn_ising_sfe_summary.json")
SPLIT_JSON = os.path.join(ROOT, "adv_datasets", "ising_sfe_split_indices.json")
CKPT_PATH = os.path.join(ROOT, "cgcnn_ising_sfe_total_energy_model.pt")
PLOT_PREFIX = os.path.join(ROOT, "cgcnn_ising_sfe")

MIN_ENERGY_EV = 1e-6
DEFAULT_SEED = 42

DEFAULT_CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
    bidirectional=True,
    aggregation="add",
    lr=2e-3,
    weight_decay=0.0,
    batch_size=32,
    huber_delta=1.0,
    scale_eps=1.0,  # absolute energies are O(10..1000) eV
    outlier_max_abs_eV=5000.0,  # soft-downweight extreme cells only
)


@dataclass
class GraphEnergyMetrics:
    mae_eV: float
    rmse_eV: float
    median_abs_err_eV: float
    mean_abs_err_eV: float
    median_rel_pct: float
    mean_rel_pct: float
    n_graphs: int
    n_rel: int


def _material_key(data) -> str:
    mat = getattr(data, "material", None)
    if mat is not None:
        return str(mat).lower()
    folder = getattr(data, "folder", None)
    if folder is not None:
        return str(folder).split("/")[0].split("\\")[0].lower()
    return f"idx_{id(data)}"


def material_grouped_split_indices(
    dataset,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """70/15/15 split by material binary so polytypes of one system stay together."""
    groups: Dict[str, List[int]] = {}
    for idx, data in enumerate(dataset):
        groups.setdefault(_material_key(data), []).append(idx)

    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)

    n = len(dataset)
    target_train = int(train_fraction * n)
    target_val = int(val_fraction * n)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    assign = {"train": [], "val": [], "test": []}

    for key in keys:
        idxs = groups[key]
        if len(train_idx) < target_train:
            train_idx.extend(idxs)
            assign["train"].append(key)
        elif len(val_idx) < target_val:
            val_idx.extend(idxs)
            assign["val"].append(key)
        else:
            test_idx.extend(idxs)
            assign["test"].append(key)

    if not val_idx or not test_idx:
        # Degenerate tiny subset — fall back to random graph split.
        perm = rng.permutation(n)
        n_train = int(train_fraction * n)
        n_val = int(val_fraction * n)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train : n_train + n_val].tolist()
        test_idx = perm[n_train + n_val :].tolist()
        assign = {"train": ["<random>"], "val": ["<random>"], "test": ["<random>"]}

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
        assign,
    )


def graph_target(data) -> torch.Tensor:
    return data.y.view(1)


def sanitize_dataset_for_batching(dataset) -> None:
    """Make graph attribute sets uniform for PyG ``Batch`` collate.

    Two failure modes from the Ising/SFE ``.pt``:

    1. Top-level ``sfe_mjm2`` missing on some graphs.
    2. ``data.meta`` dicts with *different keys* (PyG collates dicts
       recursively → ``KeyError: 'sfe_mjm2'`` inside ``meta``).

    We drop ``meta`` (not needed for training) and fill optional tensors.
    """
    for data in dataset:
        # Variable-key dicts cannot be batched — remove before DataLoader.
        if "meta" in data:
            del data.meta
        elif hasattr(data, "meta"):
            try:
                delattr(data, "meta")
            except Exception:
                pass

        data.sfe_mjm2 = (
            data.sfe_mjm2.view(-1)[:1].float()
            if ("sfe_mjm2" in data and data.sfe_mjm2 is not None)
            else torch.tensor([float("nan")], dtype=torch.float)
        )
        data.cohesive_energy_eV = (
            data.cohesive_energy_eV.view(-1)[:1].float()
            if ("cohesive_energy_eV" in data and data.cohesive_energy_eV is not None)
            else torch.tensor([float("nan")], dtype=torch.float)
        )
        data.is_faulted = bool(getattr(data, "is_faulted", False))

        for key in (
            "material",
            "kind",
            "phase",
            "fault",
            "polytype",
            "folder",
            "element_a",
            "element_b",
            "stack_sequence",
            "target_mode",
        ):
            val = getattr(data, key, None)
            if val is None:
                setattr(data, key, "")
            else:
                setattr(data, key, str(val))



def summarize_split(name: str, subset) -> None:
    if not subset:
        print(f"{name}: 0 graphs")
        return
    targets = torch.stack([graph_target(d) for d in subset]).view(-1)
    nodes = [int(d.num_nodes) for d in subset]
    mats = {_material_key(d) for d in subset}
    print(
        f"{name}: graphs={len(subset)}, materials={len(mats)}, "
        f"nodes/graph mean={np.mean(nodes):.1f}, "
        f"E_tot mean={targets.mean():.4f} eV, "
        f"std={targets.std(unbiased=False):.4f} eV",
        flush=True,
    )


def _sample_weights(targets: torch.Tensor, max_abs_eV: float) -> torch.Tensor:
    w = torch.ones_like(targets)
    if max_abs_eV > 0:
        w[targets.abs() > max_abs_eV] = 0.25
    return w


def energy_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale_eps: float,
    huber_delta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Huber on relative error (same spirit as graph-total residual trainer)."""
    pred = pred.view(-1)
    targets = targets.view(-1)
    scale = targets.abs().clamp(min=scale_eps)
    rel = (pred - targets) / scale
    loss = F.smooth_l1_loss(rel, torch.zeros_like(rel), beta=huber_delta, reduction="none")
    if weights is not None:
        loss = loss * weights.view(-1)
        return loss.sum() / weights.sum().clamp(min=1e-6)
    return loss.mean()


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> GraphEnergyMetrics:
    model.eval()
    abs_errs: List[float] = []
    sq_errs: List[float] = []
    rel_errs: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1) * target_std + target_mean
        true = batch.y.view(-1)
        err = pred - true
        for e, t in zip(err.cpu().tolist(), true.cpu().tolist()):
            abs_errs.append(abs(e))
            sq_errs.append(e * e)
            if abs(t) >= MIN_ENERGY_EV:
                rel_errs.append(100.0 * abs(e) / abs(t))

    if not abs_errs:
        return GraphEnergyMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    return GraphEnergyMetrics(
        mae_eV=float(np.mean(abs_errs)),
        rmse_eV=float(np.sqrt(np.mean(sq_errs))),
        median_abs_err_eV=float(np.median(abs_errs)),
        mean_abs_err_eV=float(np.mean(abs_errs)),
        median_rel_pct=float(np.median(rel_errs)) if rel_errs else float("nan"),
        mean_rel_pct=float(np.mean(rel_errs)) if rel_errs else float("nan"),
        n_graphs=len(abs_errs),
        n_rel=len(rel_errs),
    )


def _plot_results(
    payload: dict,
    prefix: str,
    title: str = "Ising/SFE graph-level total energy CGCNN",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plots")
        return

    run = payload["runs"]["ising_sfe"]
    epochs = range(1, len(run["val_mae_eV"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, run["train_mae_eV"], label="train")
    axes[0, 0].plot(epochs, run["val_mae_eV"], label="val")
    axes[0, 0].plot(epochs, run["test_mae_eV"], label="test")
    axes[0, 0].set_ylabel("MAE (eV)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(epochs, run["train_rel_median"], label="train")
    axes[0, 1].plot(epochs, run["val_rel_median"], label="val")
    axes[0, 1].plot(epochs, run["test_rel_median"], label="test")
    axes[0, 1].set_ylabel("Median rel. error (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    preds = run.get("test_predictions", [])
    if preds:
        true = np.array([p["true_eV"] for p in preds])
        pred = np.array([p["pred_eV"] for p in preds])
        rel = np.array([p["rel_pct"] for p in preds if p["rel_pct"] == p["rel_pct"]])
        lim_lo = float(min(true.min(), pred.min()))
        lim_hi = float(max(true.max(), pred.max()))
        axes[1, 0].scatter(true, pred, s=8, alpha=0.35, edgecolors="none")
        axes[1, 0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.6)
        axes[1, 0].set_xlabel(r"True $E_{\mathrm{tot}}$ (eV)")
        axes[1, 0].set_ylabel(r"Pred $E_{\mathrm{tot}}$ (eV)")
        axes[1, 0].grid(True, alpha=0.25)
        if len(rel):
            axes[1, 1].hist(rel, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
            axes[1, 1].axvline(np.median(rel), color="crimson", ls="--", label="median")
            axes[1, 1].set_xlabel("Test relative error (%)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.25)
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    out = f"{prefix}_epoch_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
    train_idx, val_idx, test_idx, assign = material_grouped_split_indices(dataset, seed)

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

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    print(
        f"\n=== ising_sfe | graph-level total_energy_eV | "
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

    model = build_gated_graph_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
        bidirectional=config["bidirectional"],
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
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
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
                f"[ising_sfe] ep {epoch:04d} | val MAE {val_m.mae_eV:.4f} eV | "
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
            "architecture": "gated_graph_total",
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
    print(f"[ising_sfe] Saved checkpoint -> {ckpt_path}", flush=True)

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
            rel = 100.0 * abs(err) / abs(true) if abs(true) >= MIN_ENERGY_EV else float("nan")
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
        "n_materials_train": len(assign["train"]),
        "n_materials_val": len(assign["val"]),
        "n_materials_test": len(assign["test"]),
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
            "Train gated CGCNN graph regressor on Ising/SFE total-energy graphs."
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
            "Place ising_sfe_cycle34_total_energy_dataset.pt under adv_datasets/ "
            "(or pass --dataset)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {os.path.abspath(args.dataset)}")
    print(f"Split: material-grouped 70/15/15 seed={args.seed}")

    dataset = torch.load(args.dataset, weights_only=False)
    print(f"Loaded {len(dataset)} graphs", flush=True)
    sanitize_dataset_for_batching(dataset)
    print("Sanitized optional attrs for PyG batching (sfe_mjm2/is_faulted/…)", flush=True)

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
        "config": config,
        "runs": {"ising_sfe": run},
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    summary = {
        "target": "total_energy_eV (graph-level)",
        "model": "GatedGNNGraphRegressor (pooled scalar output)",
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
        _plot_results(payload, args.plot_prefix)


if __name__ == "__main__":
    main()
