import json
import os
import time

import torch
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


def train_baseline(
    dataset_path: str,
    metric: str = "mae",
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
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(dataset_path, weights_only=False)
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

    train_curve = []
    val_curve = []
    test_curve = []
    best_val = float("inf")
    best_test = float("inf")

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
        train_curve.append(train_score)
        val_curve.append(val_score)
        test_curve.append(test_score)

        if val_score < best_val:
            best_val = val_score
            best_test = test_score

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

    return {
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "final_test": test_curve[-1] if test_curve else 0.0,
        "best_val": best_val,
        "best_test": best_test,
    }


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    ablation_json = os.path.join(root, "ablation_results.json")
    dataset_path = os.path.join(root, "pyg_dataset.pt")
    output_json = os.path.join(root, "ablation_results_with_baseline.json")
    output_plot = os.path.join(root, "ablation_results_with_baseline.png")

    with open(ablation_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    metric = payload.get("metric", "mae")

    print("\n=== Training baseline on full dataset ===")
    baseline_results = train_baseline(dataset_path, metric=metric)

    payload = dict(payload)
    results = dict(payload.get("results", {}))
    results["baseline"] = baseline_results
    payload["results"] = results

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved merged results to {output_json}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it and rerun.") from exc

    names = ["baseline"] + sorted(n for n in results.keys() if n != "baseline")
    plt.figure(figsize=(10, 5))
    for name in names:
        curve = results[name].get("test_curve")
        if not isinstance(curve, list):
            continue
        plt.plot(curve, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE without certain features")
    plt.ylim(0.0, 0.5)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
