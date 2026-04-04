"""Plot test MAE curves and per-epoch runtime from bidir_vs_gated_cycle34.json."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves from train_bidir_vs_gated_cycle34.py output JSON."
    )
    parser.add_argument("--input", type=str, default="bidir_vs_gated_cycle34.json")
    parser.add_argument("--output", type=str, default="bidir_vs_gated_cycle34_plot.png")
    parser.add_argument(
        "--curve",
        type=str,
        default="test",
        choices=["test", "val", "train", "all"],
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metric = str(payload.get("metric", "mae")).upper()
    models: Dict[str, Dict[str, List[float]]] = payload["models"]

    labels = {
        "bidirectional_base": "Bidirectional EdgeFNNConv (base)",
        "bidirectional_gated": "Bidirectional GatedConv (CGCNN)",
    }
    colors = {"bidirectional_base": "C0", "bidirectional_gated": "C1"}
    order = ["bidirectional_base", "bidirectional_gated"]

    def plot_mae(ax, curve_name: str, title: str | None = None) -> None:
        for key in order:
            series = models[key].get(curve_name, [])
            if not series:
                continue
            xs = range(1, len(series) + 1)
            ax.plot(xs, series, label=labels[key], color=colors[key], linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
        if title:
            ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_epoch_times(ax) -> None:
        for key in order:
            times = models[key].get("epoch_times", [])
            if not times:
                continue
            xs = range(1, len(times) + 1)
            ax.plot(xs, times, label=labels[key], color=colors[key], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch time (s)")
        ax.set_title("Per-epoch wall-clock time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def summary_lines() -> List[str]:
        lines = []
        for key in order:
            m = models[key]
            final_test = m["test"][-1] if m.get("test") else float("nan")
            total_t = m.get("total_train_time", 0)
            n_params = m.get("num_parameters", "?")
            lines.append(
                f"{labels[key]:42s}  test {metric} = {final_test:.4f}  |  "
                f"time = {total_t:.0f}s  |  params = {n_params:,}"
            )
        return lines

    lines = summary_lines()
    bottom_frac = 0.05 + 0.022 * max(len(lines), 1)

    if args.curve == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_mae(axes[0][0], "train", title=f"Train {metric}")
        plot_mae(axes[0][1], "val", title=f"Val {metric}")
        plot_mae(axes[1][0], "test", title=f"Test {metric}")
        plot_epoch_times(axes[1][1])
        fig.text(
            0.5, 0.01, "\n".join(lines),
            ha="center", va="bottom", fontsize=7, family="monospace",
        )
        fig.suptitle("Bidirectional EdgeFNNConv vs GatedConv (cycle34)", fontsize=13)
        fig.tight_layout(rect=[0, bottom_frac, 1, 0.96])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_mae(axes[0], args.curve, title=f"{args.curve.capitalize()} {metric}")
        plot_epoch_times(axes[1])
        fig.text(
            0.5, 0.01, "\n".join(lines),
            ha="center", va="bottom", fontsize=8, family="monospace",
        )
        fig.suptitle("Bidirectional EdgeFNNConv vs GatedConv (cycle34)", fontsize=13)
        fig.tight_layout(rect=[0, bottom_frac, 1, 0.94])

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
