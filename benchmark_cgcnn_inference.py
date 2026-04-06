"""Benchmark CGCNN GatedConv inference runtime on the full dataset.

Loads cgcnn_model.pt, runs inference on every graph individually, and tracks
runtime as a function of graph size (number of nodes).  Also reports aggregate
throughput statistics.  Outputs a JSON with per-graph timings and a plot.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import set_seed

DATASET_PATH = os.path.join("adv_datasets", "cycle34_dataset.pt")
MODEL_PATH = "cgcnn_model.pt"
OUTPUT_JSON = "cgcnn_inference_benchmark.json"
OUTPUT_PLOT = "cgcnn_inference_benchmark_plot.png"
WARMUP = 10
REPEATS = 50


def _time_single_graph(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    repeats: int,
) -> float:
    """Return mean inference time in seconds for a single graph."""
    data = data.to(device)
    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return float(np.mean(times))


def _time_batched(
    model: torch.nn.Module,
    dataset: list,
    device: torch.device,
    batch_size: int,
    repeats: int,
) -> List[float]:
    """Return per-pass times (seconds) over the full dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            for batch in loader:
                batch = batch.to(device)
                _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for batch in loader:
                batch = batch.to(device)
                _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    if not os.path.isfile(DATASET_PATH):
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = torch.load(DATASET_PATH, weights_only=False)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    cfg = checkpoint["config"]

    print(f"Dataset: {len(dataset)} graphs")
    print(f"Model config: {cfg}")
    print(f"Warmup: {WARMUP}, Repeats (batched): {REPEATS}")

    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg["use_batch_norm"],
        activation=cfg["activation"],
        bidirectional=cfg.get("bidirectional", True),
    ).to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in checkpoint["model_state"].items()}
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # --- Per-graph timing by size ---
    print("Timing individual graphs by size...")
    per_graph_repeats = 20
    graph_results: List[Dict] = []
    for i, data in enumerate(dataset):
        n_nodes = data.num_nodes
        n_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0
        mean_t = _time_single_graph(model, data, device, per_graph_repeats)
        graph_results.append({
            "index": i,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "mean_time_s": mean_t,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dataset)} graphs timed")

    print(f"  Done: {len(dataset)} graphs\n")

    # Group by node count
    by_size: Dict[int, List[float]] = defaultdict(list)
    for r in graph_results:
        by_size[r["num_nodes"]].append(r["mean_time_s"])

    print(f"{'Nodes':>6}  {'Count':>5}  {'Mean (ms)':>10}  {'Std (ms)':>9}  "
          f"{'Min (ms)':>9}  {'Max (ms)':>9}")
    print("-" * 60)
    for n_nodes in sorted(by_size.keys()):
        ts = np.array(by_size[n_nodes]) * 1000
        print(f"{n_nodes:6d}  {len(ts):5d}  {ts.mean():10.3f}  {ts.std():9.3f}  "
              f"{ts.min():9.3f}  {ts.max():9.3f}")

    # --- Batched throughput ---
    print("\nBatched throughput (batch_size=32)...")
    batched_times = _time_batched(model, dataset, device, batch_size=32, repeats=REPEATS)
    bt = np.array(batched_times) * 1000
    total_nodes = sum(d.num_nodes for d in dataset)
    print(f"  Full pass: {bt.mean():.2f} ± {bt.std():.2f} ms")
    print(f"  Throughput: {len(dataset) / (bt.mean() / 1000):.0f} graphs/s, "
          f"{total_nodes / (bt.mean() / 1000):.0f} nodes/s")

    # --- Save JSON ---
    output = {
        "device": str(device),
        "model_path": MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "num_graphs": len(dataset),
        "total_nodes": total_nodes,
        "num_parameters": n_params,
        "per_graph_repeats": per_graph_repeats,
        "batched_repeats": REPEATS,
        "per_graph": graph_results,
        "by_node_count": {
            str(k): {
                "count": len(v),
                "mean_ms": float(np.mean(v) * 1000),
                "std_ms": float(np.std(v) * 1000),
            }
            for k, v in sorted(by_size.items())
        },
        "batched_pass_times_ms": [float(t) for t in bt],
        "batched_mean_ms": float(bt.mean()),
        "batched_std_ms": float(bt.std()),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {OUTPUT_JSON}")

    # --- Plot ---
    nodes_arr = np.array([r["num_nodes"] for r in graph_results])
    times_arr = np.array([r["mean_time_s"] for r in graph_results]) * 1000
    edges_arr = np.array([r["num_edges"] for r in graph_results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter: time vs nodes
    ax = axes[0]
    ax.scatter(nodes_arr, times_arr, s=12, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference time vs graph size (nodes)")
    ax.grid(True, alpha=0.3)

    # 2. Scatter: time vs edges
    ax = axes[1]
    ax.scatter(edges_arr, times_arr, s=12, alpha=0.5, color="C1", edgecolors="none")
    ax.set_xlabel("Number of edges")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference time vs graph size (edges)")
    ax.grid(True, alpha=0.3)

    # 3. Box plot by node count
    ax = axes[2]
    sorted_sizes = sorted(by_size.keys())
    box_data = [np.array(by_size[s]) * 1000 for s in sorted_sizes]
    bp = ax.boxplot(box_data, labels=[str(s) for s in sorted_sizes],
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.5)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Runtime distribution by graph size")
    ax.grid(True, alpha=0.3, axis="y")

    footer = (
        f"CGCNN GatedConv (hd=128, nl=2, bidir) | {n_params:,} params | "
        f"{len(dataset)} graphs, {total_nodes} total nodes | "
        f"batched throughput: {len(dataset) / (bt.mean() / 1000):.0f} graphs/s"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, family="monospace")
    fig.suptitle("CGCNN inference runtime analysis", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
