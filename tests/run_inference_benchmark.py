from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and benchmark runtime.")
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--checkpoint", type=str, default="base_model.pt")
    parser.add_argument("--output", type=str, default="inference_timing.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, weights_only=False)

    model, target_mean, target_std = _load_model(args.checkpoint, dataset, device)
    target_mean = torch.tensor(target_mean, device=device)
    target_std = torch.tensor(target_std, device=device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Warmup
    warmup_steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch) * target_std + target_mean
            warmup_steps += 1
            if warmup_steps >= args.warmup:
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

    # Also provide coarse bins for readability
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 9999)]
    per_bin = {}
    for lo, hi in bins:
        values = [t for n, t in zip(node_counts, timing_ms) if lo <= n <= hi]
        if values:
            per_bin[f"{lo}-{hi}"] = {
                "count": len(values),
                "avg_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
            }

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "overall": overall,
        "per_node_count": per_node,
        "per_node_bin": per_bin,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved timing report to {args.output}")


if __name__ == "__main__":
    main()
