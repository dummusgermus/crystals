"""Plot test MAE curves and per-epoch runtime from bidir_vs_towers_cycle34.json."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves from train_bidir_vs_towers_cycle34.py output JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="bidir_vs_towers_cycle34.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bidir_vs_towers_cycle34_plot.png",
    )
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
        "bidirectional": "Bidirectional base",
        "bidirectional_towers": "Bidirectional + 8 towers",
    }
    colors = {"bidirectional": "C0", "bidirectional_towers": "C1"}

    def plot_mae(ax, curve_name: str, title: str | None = None) -> None:
        for key in ("bidirectional", "bidirectional_towers"):
            series = models[key].get(curve_name, [])
            if not series:
                continue
            xs = range(1, len(series) + 1)
            ax.plot(xs, series, label=labels[key], color=colors[key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_epoch_times(ax) -> None:
        for key in ("bidirectional", "bidirectional_towers"):
            times = models[key].get("epoch_times", [])
            if not times:
                continue
            xs = range(1, len(times) + 1)
            ax.plot(xs, times, label=labels[key], color=colors[key], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch time (s)")
        ax.set_title("Per-epoch wall-clock time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def add_summary_text(fig) -> None:
        lines = []
        for key in ("bidirectional", "bidirectional_towers"):
            m = models[key]
            final_test = m["test"][-1] if m["test"] else float("nan")
            total_t = m.get("total_train_time", 0)
            n_params = m.get("num_parameters", "?")
            lines.append(
                f"{labels[key]}:  test {metric} = {final_test:.4f}  |  "
                f"time = {total_t:.0f}s  |  params = {n_params:,}"
            )
        fig.text(
            0.5, 0.01, "\n".join(lines),
            ha="center", va="bottom", fontsize=8, family="monospace",
        )

    if args.curve == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, name in zip(axes[0], ("train", "val")):
            plot_mae(ax, name, title=f"{name.capitalize()} {metric}")
        plot_mae(axes[1][0], "test", title=f"Test {metric}")
        plot_epoch_times(axes[1][1])
        add_summary_text(fig)
        fig.suptitle("Bidirectional base vs multi-tower MPNN (cycle34)", fontsize=13)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_mae(axes[0], args.curve, title=f"{args.curve.capitalize()} {metric}")
        plot_epoch_times(axes[1])
        add_summary_text(fig)
        fig.suptitle("Bidirectional base vs multi-tower MPNN (cycle34)", fontsize=13)
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
