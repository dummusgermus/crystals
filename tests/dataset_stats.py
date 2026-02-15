import os
from itertools import product
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch

import graph_maker


def _counts(dataset: Iterable) -> Tuple[List[int], List[int]]:
    node_counts = []
    edge_counts = []
    for data in dataset:
        node_counts.append(int(data.num_nodes))
        edge_counts.append(int(data.edge_index.shape[1]))
    return node_counts, edge_counts


def _summarize_counts(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": float(sum(counts)) / float(len(counts)),
    }


def _print_summary(title: str, node_counts: List[int], edge_counts: List[int]) -> None:
    node_stats = _summarize_counts(node_counts)
    edge_stats = _summarize_counts(edge_counts)
    print(title)
    print(f"  graphs: {len(node_counts)}")
    print(
        "  nodes  min/max/mean: "
        f"{node_stats['min']} / {node_stats['max']} / {node_stats['mean']:.2f}"
    )
    print(
        "  edges  min/max/mean: "
        f"{edge_stats['min']} / {edge_stats['max']} / {edge_stats['mean']:.2f}"
    )


def plot_histograms(
    node_counts: List[int], edge_counts: List[int], output_path: str
) -> None:
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(node_counts, bins=30, color="#4C78A8", alpha=0.8)
    axes[0].set_title("Node Count Histogram")
    axes[0].set_xlabel("Nodes per graph")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(edge_counts, bins=30, color="#F58518", alpha=0.8)
    axes[1].set_title("Edge Count Histogram")
    axes[1].set_xlabel("Edges per graph")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_and_summarize(
    simulations_dir: str,
    cutoff_k: int,
    edge_k: int,
    cutoff_radius: float,
    edge_radius: float,
    cutoff_mode: str,
) -> Tuple[List[int], List[int]]:
    dataset = graph_maker.build_pyg_dataset(
        simulations_dir=simulations_dir,
        cutoff_k=cutoff_k,
        edge_k=edge_k,
        cutoff_radius=cutoff_radius,
        edge_radius=edge_radius,
        cutoff_mode=cutoff_mode,
    )
    node_counts, edge_counts = _counts(dataset)
    _print_summary(
        f"cutoff_k={cutoff_k}, edge_k={edge_k}",
        node_counts,
        edge_counts,
    )
    return node_counts, edge_counts


def plot_sweep_averages(
    sweep_results: Dict[Tuple[float, float], Dict[str, float]],
    output_path: str,
    title_prefix: str,
    x_label: str,
    edge_label: str,
) -> None:
    matplotlib.use("Agg")
    cutoff_values = sorted({k[0] for k in sweep_results})
    edge_values = sorted({k[1] for k in sweep_results})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for edge_k in edge_values:
        node_means = [
            sweep_results[(cutoff_k, edge_k)]["nodes_mean"]
            for cutoff_k in cutoff_values
        ]
        edge_means = [
            sweep_results[(cutoff_k, edge_k)]["edges_mean"]
            for cutoff_k in cutoff_values
        ]
        axes[0].plot(
            cutoff_values,
            node_means,
            marker="o",
            label=f"{edge_label}={edge_k}",
        )
        axes[1].plot(
            cutoff_values,
            edge_means,
            marker="o",
            label=f"{edge_label}={edge_k}",
        )

    axes[0].set_title(f"{title_prefix} Average Node Count")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("avg nodes per graph")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(f"{title_prefix} Average Edge Count")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("avg edges per graph")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_sweep_txt(
    sweep_results: Dict[Tuple[int, int], Dict[str, float]],
    output_path: str,
    title_prefix: str,
    mode: str,
) -> None:
    lines = []
    lines.append(f"{title_prefix} sweep")
    lines.append("cutoff_k,edge_k,graphs,nodes_min,nodes_max,nodes_mean,edges_min,edges_max,edges_mean")
    for (cutoff_k, edge_k) in sorted(sweep_results):
        row = sweep_results[(cutoff_k, edge_k)]
        lines.append(
            f"{cutoff_k},{edge_k},{row['graphs']},"
            f"{row['nodes_min']},{row['nodes_max']},{row['nodes_mean']:.6f},"
            f"{row['edges_min']},{row['edges_max']},{row['edges_mean']:.6f}"
        )
    with open(output_path, mode, encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    simulations_dir = os.path.join(root_dir, "SIMULATIONS")
    dataset_path = os.path.join(root_dir, "pyg_dataset.pt")
    histogram_path = os.path.join(root_dir, "pyg_dataset_hist.png")
    sweep_plot_path_shell = os.path.join(root_dir, "pyg_sweep_shell.png")
    sweep_plot_path_radius = os.path.join(root_dir, "pyg_sweep_radius.png")
    sweep_txt_path = os.path.join(root_dir, "pyg_sweep_stats.txt")

    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path, weights_only=False)
        node_counts, edge_counts = _counts(dataset)
        _print_summary("pyg_dataset.pt summary", node_counts, edge_counts)
        plot_histograms(node_counts, edge_counts, histogram_path)
        print(f"Saved histogram to {histogram_path}")
    else:
        print("pyg_dataset.pt not found, skipping histogram step.")

    shell_cutoff_values = [3, 4, 5, 6, 7, 8]
    shell_edge_values = [1, 2, 3]
    radius_cutoff_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    radius_edge_values = [2.0, 3.0, 4.0]

    print("\nDataset size sweep (shell mode):")
    sweep_results_shell: Dict[Tuple[int, int], Dict[str, float]] = {}
    for cutoff_k, edge_k in product(shell_cutoff_values, shell_edge_values):
        node_counts, edge_counts = build_and_summarize(
            simulations_dir,
            cutoff_k,
            edge_k,
            cutoff_radius=graph_maker.DEFECT_CUTOFF_RADIUS,
            edge_radius=graph_maker.EDGE_CUTOFF_RADIUS,
            cutoff_mode="shell",
        )
        node_stats = _summarize_counts(node_counts)
        edge_stats = _summarize_counts(edge_counts)
        sweep_results_shell[(cutoff_k, edge_k)] = {
            "graphs": len(node_counts),
            "nodes_min": node_stats["min"],
            "nodes_max": node_stats["max"],
            "nodes_mean": node_stats["mean"],
            "edges_min": edge_stats["min"],
            "edges_max": edge_stats["max"],
            "edges_mean": edge_stats["mean"],
        }

    print("\nDataset size sweep (radius mode):")
    sweep_results_radius: Dict[Tuple[int, int], Dict[str, float]] = {}
    for cutoff_k, edge_k in product(radius_cutoff_values, radius_edge_values):
        node_counts, edge_counts = build_and_summarize(
            simulations_dir,
            int(round(cutoff_k)),
            int(round(edge_k)),
            cutoff_radius=float(cutoff_k),
            edge_radius=float(edge_k),
            cutoff_mode="radius",
        )
        node_stats = _summarize_counts(node_counts)
        edge_stats = _summarize_counts(edge_counts)
        sweep_results_radius[(cutoff_k, edge_k)] = {
            "graphs": len(node_counts),
            "nodes_min": node_stats["min"],
            "nodes_max": node_stats["max"],
            "nodes_mean": node_stats["mean"],
            "edges_min": edge_stats["min"],
            "edges_max": edge_stats["max"],
            "edges_mean": edge_stats["mean"],
        }

    plot_sweep_averages(
        sweep_results_shell, sweep_plot_path_shell, "Shell", "cutoff_k", "edge_k"
    )
    plot_sweep_averages(
        sweep_results_radius, sweep_plot_path_radius, "Radius", "cutoff_r (Å)", "edge_r (Å)"
    )
    write_sweep_txt(sweep_results_shell, sweep_txt_path, "Shell", mode="w")
    write_sweep_txt(sweep_results_radius, sweep_txt_path, "Radius", mode="a")
    print(f"Saved sweep plot to {sweep_plot_path_shell}")
    print(f"Saved sweep plot to {sweep_plot_path_radius}")
    print(f"Saved sweep stats to {sweep_txt_path}")


if __name__ == "__main__":
    main()
