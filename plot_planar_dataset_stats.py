"""Plot summary statistics for the planar PyG dataset.

Loads planar_pyg_dataset.pt and writes a multi-panel figure with
distributions of graph size, connectivity, stack sequences, targets, etc.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DATASET = "planar_pyg_dataset.pt"
DEFAULT_OUTPUT = "planar_dataset_stats.png"
DEFAULT_JSON = "planar_dataset_stats.json"

# Canonical stack order (shortest to longest supercell).
STACK_ORDER = [
    "XY'",
    "XY'X",
    "XY'XY",
    "XYZ'Y'",
    "XYZ'Y'X'",
    "XYZ'Y'X'Z'",
    "XY'XYZXY",
    "XYZ'Y'X'Z'Y'X'Z'",
    "XY'XYZXYZXYZX",
    "XYZ'Y'X'Z'Y'X'Z'Y'X'Z'Y'X'",
]


def _stack_sort_key(stack: str) -> int:
    try:
        return STACK_ORDER.index(stack)
    except ValueError:
        return len(STACK_ORDER)


def collect_statistics(dataset) -> Dict:
    node_counts: List[int] = []
    edge_counts: List[int] = []
    mean_pe_per_graph: List[float] = []
    type1_fraction: List[float] = []
    degrees: List[int] = []
    nodes_by_stack: Dict[str, List[int]] = defaultdict(list)
    edges_by_stack: Dict[str, List[int]] = defaultdict(list)
    chem_systems: Counter = Counter()
    stacks: Counter = Counter()
    has_unrelaxed_pe = 0

    for data in dataset:
        n = int(data.num_nodes)
        e = int(data.edge_index.size(1))
        node_counts.append(n)
        edge_counts.append(e)

        y = data.y.view(-1).cpu().numpy()
        mean_pe_per_graph.append(float(np.mean(y)))

        types = data.x[:, 0].cpu().numpy()
        type1_fraction.append(float(np.mean(types == 1.0)))

        adj = defaultdict(int)
        ei = data.edge_index.cpu().numpy()
        for src, dst in zip(ei[0], ei[1]):
            adj[int(src)] += 1
            adj[int(dst)] += 1
        degrees.extend(adj.values())

        stack = str(getattr(data, "stack_sequence", "unknown"))
        stacks[stack] += 1
        nodes_by_stack[stack].append(n)
        edges_by_stack[stack].append(e)

        elem_a = getattr(data, "element_a", None)
        elem_b = getattr(data, "element_b", None)
        if elem_a and elem_b:
            chem_systems[f"{elem_a}-{elem_b}"] += 1

        meta = getattr(data, "meta", None) or {}
        if meta.get("has_unrelaxed_pe"):
            has_unrelaxed_pe += 1

    return {
        "num_graphs": len(dataset),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "mean_pe_per_graph": mean_pe_per_graph,
        "type1_fraction": type1_fraction,
        "degrees": degrees,
        "nodes_by_stack": dict(nodes_by_stack),
        "edges_by_stack": dict(edges_by_stack),
        "stacks": dict(stacks),
        "chem_systems": dict(chem_systems),
        "graphs_with_unrelaxed_pe": has_unrelaxed_pe,
        "unique_node_counts": sorted(set(node_counts)),
        "unique_edge_counts": sorted(set(edge_counts)),
    }


def _summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def plot_statistics(stats: Dict, output_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Planar defect dataset statistics", fontsize=14)

    # 1) Node count distribution
    ax = axes[0, 0]
    unique_nodes, node_freq = np.unique(stats["node_counts"], return_counts=True)
    ax.bar(unique_nodes, node_freq, width=4, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Nodes per graph")
    ax.set_ylabel("Number of graphs")
    ax.set_title("Node count distribution")
    ax.set_xticks(unique_nodes)

    # 2) Edge count distribution
    ax = axes[0, 1]
    ax.hist(stats["edge_counts"], bins=20, color="#F58518", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Edges per graph")
    ax.set_ylabel("Number of graphs")
    ax.set_title("Edge count distribution")

    # 3) Mean relaxed PE per graph
    ax = axes[0, 2]
    ax.hist(stats["mean_pe_per_graph"], bins=30, color="#54A24B", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean relaxed per-atom PE (eV)")
    ax.set_ylabel("Number of graphs")
    ax.set_title("Target PE per graph")

    # 4) Nodes by stack sequence
    ax = axes[1, 0]
    stack_keys = sorted(stats["stacks"].keys(), key=_stack_sort_key)
    stack_nodes = [
        float(np.mean(stats["nodes_by_stack"][s])) for s in stack_keys
    ]
    xpos = np.arange(len(stack_keys))
    ax.bar(xpos, stack_nodes, color="#B279A2", alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(stack_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Nodes per graph")
    ax.set_title("Mean node count by stack sequence")

    # 5) Node degree distribution (pooled)
    ax = axes[1, 1]
    ax.hist(stats["degrees"], bins=range(0, max(stats["degrees"]) + 2), color="#E45756", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Count (all nodes)")
    ax.set_title("Degree distribution")

    # 6) Chemistry coverage
    ax = axes[1, 2]
    chem_counts = Counter(stats["chem_systems"])
    top_n = 12
    top = chem_counts.most_common(top_n)
    labels = [c[0] for c in top]
    counts = [c[1] for c in top]
    ypos = np.arange(len(labels))
    ax.barh(ypos, counts, color="#72B7B2", alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Graphs per chemistry")
    ax.set_title(
        f"Top {top_n} chemistries "
        f"({len(chem_counts)} unique, 10 stacks each)"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_json_summary(stats: Dict, output_path: str) -> None:
    summary = {
        "num_graphs": stats["num_graphs"],
        "unique_chemistries": len(stats["chem_systems"]),
        "unique_stack_sequences": len(stats["stacks"]),
        "graphs_with_unrelaxed_pe": stats["graphs_with_unrelaxed_pe"],
        "node_count": _summary_stats(stats["node_counts"]),
        "edge_count": _summary_stats(stats["edge_counts"]),
        "mean_pe_per_graph": _summary_stats(stats["mean_pe_per_graph"]),
        "type1_fraction": _summary_stats(stats["type1_fraction"]),
        "degree": _summary_stats(stats["degrees"]),
        "unique_node_counts": stats["unique_node_counts"],
        "graphs_per_stack": stats["stacks"],
        "mean_nodes_by_stack": {
            s: float(np.mean(stats["nodes_by_stack"][s]))
            for s in sorted(stats["nodes_by_stack"], key=_stack_sort_key)
        },
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot summary statistics for planar_pyg_dataset.pt"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to planar PyG dataset (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output PNG path",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=DEFAULT_JSON,
        help="Optional JSON summary path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    dataset = torch.load(args.dataset, weights_only=False)
    stats = collect_statistics(dataset)
    plot_statistics(stats, args.output)
    write_json_summary(stats, args.json_output)

    print(f"Graphs: {stats['num_graphs']}")
    print(f"Node counts: {stats['unique_node_counts']}")
    print(f"Saved plot  -> {args.output}")
    print(f"Saved stats -> {args.json_output}")


if __name__ == "__main__":
    main()
