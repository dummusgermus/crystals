import json
import os
import time

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
)

NODE_FEATURES = ["type", "per_atom_pe", "is_defect", "dist_to_defect"]
EDGE_FEATURES = ["distance", "same_type", "incident_defect"]

def _drop_features(data: Data, node_idx: int | None, edge_idx: int | None) -> Data:
    new_data = data.clone()
    if node_idx is not None:
        keep = [i for i in range(new_data.x.size(1)) if i != node_idx]
        new_data.x = new_data.x[:, keep]
    if edge_idx is not None:
        keep = [i for i in range(new_data.edge_attr.size(1)) if i != edge_idx]
        new_data.edge_attr = new_data.edge_attr[:, keep]
    return new_data


def _build_ablation_dataset(dataset, node_idx: int | None, edge_idx: int | None):
    return [_drop_features(data, node_idx, edge_idx) for data in dataset]


def train_base(
    dataset,
    metric: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    use_batch_norm: bool = False,
    activation: str = "silu",
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    test_curve = []
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

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )

        train_score = metric_value(train_metrics, metric)
        val_score = metric_value(val_metrics, metric)
        test_score = metric_value(test_metrics, metric)
        test_curve.append(test_score)

        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train {metric.upper()} {train_score:.4f} | "
            f"val {metric.upper()} {val_score:.4f} | "
            f"test {metric.upper()} {test_score:.4f} | "
            f"lr {current_lr:.1e} | "
            f"time {dt:.1f}s"
        )

        scheduler.step(val_score)

    return test_curve


def _load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _feature_dims(dataset) -> tuple[int, int]:
    if not dataset:
        return 0, 0
    node_dim = int(dataset[0].x.size(1))
    edge_dim = int(dataset[0].edge_attr.size(1))
    return node_dim, edge_dim


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    base_results_path = os.path.join(root, "ablation_results_with_baseline.json")
    fallback_results_path = os.path.join(root, "ablation_results.json")
    full_dataset_path = os.path.join(root, "pyg_dataset.pt")
    combo_dataset_path = os.path.join(root, "datasets", "pyg_dataset_no_node_dist_to_defect_and_no_edge_distance.pt")
    output_plot = os.path.join(root, "ablation_selected_plot.png")

    results_path = base_results_path if os.path.exists(base_results_path) else fallback_results_path
    payload = _load_results(results_path)
    metric = payload.get("metric", "mae")
    results = payload.get("results", {})

    baseline_curve = results.get("baseline", {}).get("test_curve")
    if not isinstance(baseline_curve, list):
        raise SystemExit(
            "Missing baseline curve. Put it in ablation_results_with_baseline.json "
            "or add a 'baseline' entry to ablation_results.json."
        )

    no_dist_key = "pyg_dataset_no_node_dist_to_defect"
    no_edge_key = "pyg_dataset_no_edge_distance"
    no_dist_curve = results.get(no_dist_key, {}).get("test_curve")
    no_edge_curve = results.get(no_edge_key, {}).get("test_curve")
    if not isinstance(no_dist_curve, list) or not isinstance(no_edge_curve, list):
        raise SystemExit(
            "Missing curves for required ablation datasets in results JSON."
        )

    full_dataset = torch.load(full_dataset_path, weights_only=False)
    node_idx = NODE_FEATURES.index("dist_to_defect")
    edge_idx = EDGE_FEATURES.index("distance")
    combo_dataset = _build_ablation_dataset(full_dataset, node_idx=node_idx, edge_idx=edge_idx)
    os.makedirs(os.path.dirname(combo_dataset_path), exist_ok=True)
    torch.save(combo_dataset, combo_dataset_path)

    orig_node_dim, orig_edge_dim = _feature_dims(full_dataset)
    combo_node_dim, combo_edge_dim = _feature_dims(combo_dataset)
    if combo_node_dim != orig_node_dim - 1 or combo_edge_dim != orig_edge_dim - 1:
        raise SystemExit(
            "Combined dataset feature dimensions look wrong. "
            f"Original (nodes={orig_node_dim}, edges={orig_edge_dim}), "
            f"combined (nodes={combo_node_dim}, edges={combo_edge_dim})."
        )
    print(
        "Combined dataset saved with feature dims: "
        f"nodes {orig_node_dim}->{combo_node_dim}, "
        f"edges {orig_edge_dim}->{combo_edge_dim}"
    )

    print("\n=== Training dataset without dist_to_defect and edge_distance ===")
    combo_dataset = torch.load(combo_dataset_path, weights_only=False)
    combo_curve = train_base(combo_dataset, metric=metric)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it and rerun.") from exc

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_curve, label="baseline")
    plt.plot(no_dist_curve, label="no_dist_to_defect")
    plt.plot(no_edge_curve, label="no_edge_distance")
    plt.plot(combo_curve, label="no_dist_to_defect + no_edge_distance")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric.upper()} (test_curve)")
    plt.title(f"Ablation comparison ({metric.upper()})")
    plt.ylim(0.0, 0.5)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
