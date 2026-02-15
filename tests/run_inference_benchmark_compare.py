from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import GNNNodeRegressor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(checkpoint_path: str, dataset, device: torch.device) -> Tuple[torch.nn.Module, float, float]:
    ckpt = torch.load(checkpoint_path, weights_only=False)
    config = ckpt["config"]
    model = GNNNodeRegressor(
        in_dim=dataset[0].x.size(-1),
        edge_dim=dataset[0].edge_attr.size(-1),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, float(ckpt["target_mean"]), float(ckpt["target_std"])


def _benchmark(
    dataset,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 10,
) -> Dict:
    model, target_mean, target_std = _load_model(checkpoint_path, dataset, device)
    target_mean = torch.tensor(target_mean, device=device)
    target_std = torch.tensor(target_std, device=device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    warmup_steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch) * target_std + target_mean
            warmup_steps += 1
            if warmup_steps >= warmup:
                break

    timing_ms: List[float] = []
    node_counts: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _sync(device)
            t0 = time.perf_counter()
            _ = model(batch) * target_std + target_mean
            _sync(device)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            timing_ms.append(elapsed_ms)
            node_counts.append(int(batch.num_nodes))

    overall = {
        "graphs": len(node_counts),
        "avg_ms": float(np.mean(timing_ms)),
        "median_ms": float(np.median(timing_ms)),
        "p90_ms": float(np.percentile(timing_ms, 90)),
        "p99_ms": float(np.percentile(timing_ms, 99)),
    }

    by_nodes: Dict[int, List[float]] = defaultdict(list)
    for n, t in zip(node_counts, timing_ms):
        by_nodes[int(n)].append(float(t))

    per_node = {}
    for n, values in sorted(by_nodes.items(), key=lambda x: x[0]):
        per_node[str(n)] = {
            "count": len(values),
            "avg_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
        }

    return {
        "overall": overall,
        "per_node_count": per_node,
    }


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    base_dataset = os.path.join(root, "pyg_dataset.pt")
    base_ckpt = os.path.join(root, "base_model.pt")
    reduced_dataset = os.path.join(
        root, "datasets", "pyg_dataset_no_node_dist_to_defect_and_no_edge_distance.pt"
    )
    reduced_ckpt = os.path.join(root, "base_model_no_dist_edge.pt")
    output_json = os.path.join(root, "inference_timing_compare.json")
    output_plot = os.path.join(root, "inference_timing_compare.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_data = torch.load(base_dataset, weights_only=False)
    reduced_data = torch.load(reduced_dataset, weights_only=False)

    base_result = _benchmark(base_data, base_ckpt, device)
    reduced_result = _benchmark(reduced_data, reduced_ckpt, device)

    result = {
        "device": str(device),
        "batch_size": 1,
        "baseline_full": base_result,
        "reduced_no_dist_edge": reduced_result,
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved timing report to {output_json}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it and rerun.") from exc

    def _series(data: Dict) -> Tuple[List[int], List[float]]:
        per_node = data.get("per_node_count", {})
        node_counts = sorted(int(k) for k in per_node.keys())
        avg_ms = [per_node[str(n)]["avg_ms"] for n in node_counts]
        return node_counts, avg_ms

    base_nodes, base_avg = _series(base_result)
    red_nodes, red_avg = _series(reduced_result)

    plt.figure(figsize=(8, 5))
    plt.plot(base_nodes, base_avg, label="baseline avg", marker="o", linewidth=1.0, markersize=3)
    plt.plot(red_nodes, red_avg, label="reduced avg", marker="o", linewidth=1.0, markersize=3)
    plt.xlabel("Number of nodes in graph")
    plt.ylabel("Inference time (ms)")
    plt.title("Inference time vs node count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
