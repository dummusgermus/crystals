"""Plot train/val/test curves from base_vs_bidirectional_cycle34.json.

Optionally overlays curves from masternode_cycle34.json (train_masternode_cycle34.py).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves from train_base_vs_bidirectional_cycle34.py output JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="base_vs_bidirectional_cycle34.json",
        help="JSON file from train_base_vs_bidirectional_cycle34.py",
    )
    parser.add_argument(
        "--masternode",
        type=str,
        default="masternode_cycle34.json",
        help=(
            "Optional JSON from train_masternode_cycle34.py; if the file exists, "
            "its curves are drawn on the same axes."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_vs_bidirectional_cycle34_plot.png",
        help="Output PNG filename",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which metric series to plot (all = three subplots).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metric = str(payload.get("metric", "mae")).upper()
    models: Dict[str, Dict[str, List[float]]] = payload["models"]

    masternode_curves: Optional[Dict[str, List[float]]] = None
    if args.masternode and os.path.isfile(args.masternode):
        with open(args.masternode, "r", encoding="utf-8") as f:
            mn = json.load(f)
        masternode_curves = mn.get("curves")
        if masternode_curves is None:
            print(f"Warning: {args.masternode} has no 'curves' key; skipping overlay.")
            masternode_curves = None
    elif args.masternode:
        print(f"Note: {args.masternode} not found; plotting base vs bidirectional only.")

    labels = {
        "base": "Undirected",
        "bidirectional": "Directed",
    }
    colors = {"base": "C0", "bidirectional": "C1"}

    def plot_series(ax, curve_name: str, title: str | None = None) -> None:
        for key in ("base", "bidirectional"):
            if key not in models:
                continue
            series = models[key].get(curve_name, [])
            if not series:
                continue
            xs = range(1, len(series) + 1)
            ax.plot(
                xs,
                series,
                label=labels.get(key, key),
                color=colors.get(key),
            )
        if masternode_curves is not None:
            series = masternode_curves.get(curve_name, [])
            if series:
                xs = range(1, len(series) + 1)
                ax.plot(
                    xs,
                    series,
                    label="Directed + master node",
                    color="C2",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.013, 0.02)

    if args.curve == "all":
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for ax, name in zip(axes, ("train", "val", "test")):
            plot_series(ax, name, title=f"{name.capitalize()} {metric}")
        fig.suptitle(f"Undirected vs directed")
        fig.tight_layout()
        plt.savefig(args.output, dpi=150)
    else:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        plot_series(ax, args.curve, title=None)
        plt.title(f"Undirected vs directed MAE")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
