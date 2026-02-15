from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset

NODE_FEATURES = ["type", "per_atom_pe", "is_defect", "dist_to_defect"]
EDGE_FEATURES = ["distance", "same_type", "incident_defect"]


@dataclass
class Metrics:
    mae: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _per_graph_reduction(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    sums = torch.zeros(num_graphs, device=values.device)
    counts = torch.zeros(num_graphs, device=values.device)
    sums.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(values))
    return sums / counts.clamp(min=1.0)


def per_graph_mae_loss(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.tensor(0.0, device=pred.device)
    abs_err = torch.abs(pred - target)
    per_graph_mae = _per_graph_reduction(abs_err, batch, num_graphs)
    return per_graph_mae.mean()


def compute_mae(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> Metrics:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return Metrics(mae=0.0)
    abs_err = torch.abs(pred - target)
    per_graph_mae = _per_graph_reduction(abs_err, batch, num_graphs)
    return Metrics(mae=per_graph_mae.mean().item())


def _drop_features(data: Data, node_idx: int | None, edge_idx: int | None) -> Data:
    new_data = data.clone()
    if node_idx is not None:
        keep = [i for i in range(new_data.x.size(1)) if i != node_idx]
        new_data.x = new_data.x[:, keep]
    if edge_idx is not None:
        keep = [i for i in range(new_data.edge_attr.size(1)) if i != edge_idx]
        new_data.edge_attr = new_data.edge_attr[:, keep]
    return new_data


def _build_reduced_dataset(dataset: List[Data]) -> List[Data]:
    node_idx = NODE_FEATURES.index("dist_to_defect")
    edge_idx = EDGE_FEATURES.index("distance")
    return [_drop_features(data, node_idx=node_idx, edge_idx=edge_idx) for data in dataset]


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(
        root, "datasets", "pyg_dataset_no_node_dist_to_defect_and_no_edge_distance.pt"
    )
    model_output = os.path.join(root, "base_model_no_dist_edge.pt")

    epochs = 40
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    hidden_dim = 128
    num_layers = 3
    dropout = 0.1
    use_batch_norm = False
    activation = "silu"
    seed = 42

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(dataset_path):
        raise SystemExit(f"Reduced dataset not found: {dataset_path}")
    reduced_dataset = torch.load(dataset_path, weights_only=False)

    loader = DataLoader(reduced_dataset, batch_size=batch_size, shuffle=True)

    train_targets = torch.cat([data.y for data in reduced_dataset], dim=0).view(-1)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    model = build_model_from_dataset(
        reduced_dataset,
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

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= max(len(loader.dataset), 1)

        model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            batches = []
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch) * target_std + target_mean
                preds.append(pred)
                targets.append(batch.y)
                batches.append(batch.batch)
            pred = torch.cat(preds, dim=0)
            target = torch.cat(targets, dim=0)
            batch_idx = torch.cat(batches, dim=0)
            metrics = compute_mae(pred, target, batch_idx)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train MAE {metrics.mae:.4f} | "
            f"time {dt:.1f}s"
        )

    checkpoint: Dict = {
        "model_state": model.state_dict(),
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
            "activation": activation,
        },
        "target_mean": float(target_mean.detach().cpu().item()),
        "target_std": float(target_std.detach().cpu().item()),
        "metrics": asdict(metrics),
    }
    torch.save(checkpoint, model_output)
    print(f"Saved model to {model_output}")


if __name__ == "__main__":
    main()
