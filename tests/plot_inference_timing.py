from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot inference timing results.")
    parser.add_argument("--input", type=str, default="inference_timing.json")
    parser.add_argument("--output", type=str, default="inference_timing.png")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_node = data.get("per_node_count", {})
    if not per_node:
        raise SystemExit("No per_node_count data found in timing JSON.")

    node_counts = sorted(int(k) for k in per_node.keys())
    avg_ms = [per_node[str(n)]["avg_ms"] for n in node_counts]
    med_ms = [per_node[str(n)]["median_ms"] for n in node_counts]

    plt.figure(figsize=(8, 5))
    plt.plot(node_counts, avg_ms, label="Avg ms", marker="o", linewidth=1.0, markersize=3)
    plt.plot(node_counts, med_ms, label="Median ms", marker="o", linewidth=1.0, markersize=3)
    plt.xlabel("Number of nodes in graph")
    plt.ylabel("Inference time (ms)")
    overall = data.get("overall", {})
    title = "Inference time vs node count"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
