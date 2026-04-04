"""Analyse and plot results from the GatedConv hyperparameter sweep.

Produces a summary table of the top-N configs (by validation MAE) and a
scatter plot of val MAE vs test MAE coloured by a chosen hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [r for r in payload["results"] if "mean_val_mae" in r]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GatedConv sweep results.")
    parser.add_argument("--input", type=str, default="sweep_gated_cycle34.json")
    parser.add_argument("--output", type=str, default="sweep_gated_cycle34_plot.png")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    results = _load(args.input)
    if not results:
        raise SystemExit("No valid results found.")

    results.sort(key=lambda r: r["mean_val_mae"])
    print(f"Loaded {len(results)} valid runs\n")

    print(f"{'Rank':>4}  {'Val MAE':>9}  {'Test MAE':>9}  {'Params':>9}  "
          f"{'Time':>6}  {'hd':>4}  {'nl':>2}  {'dr':>5}  {'lr':>7}  "
          f"{'wd':>7}  {'bs':>3}  {'act':>5}  {'bn':>5}")
    print("-" * 110)
    for rank, r in enumerate(results[: args.top_n], 1):
        c = r["config"]
        print(
            f"{rank:4d}  {r['mean_val_mae']:9.5f}  {r['mean_test_mae']:9.5f}  "
            f"{r['num_parameters']:9,}  {r['total_train_time']:5.0f}s  "
            f"{c['hidden_dim']:4d}  {c['num_layers']:2d}  {c['dropout']:5.2f}  "
            f"{c['lr']:7.1e}  {c['weight_decay']:7.1e}  {c['batch_size']:3d}  "
            f"{c['activation']:>5}  {str(c['use_batch_norm']):>5}"
        )

    val_maes = np.array([r["mean_val_mae"] for r in results])
    test_maes = np.array([r["mean_test_mae"] for r in results])
    hidden_dims = np.array([r["config"]["hidden_dim"] for r in results])
    num_layers = np.array([r["config"]["num_layers"] for r in results])
    times = np.array([r["total_train_time"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Val vs Test MAE scatter, coloured by hidden_dim
    ax = axes[0][0]
    sc = ax.scatter(val_maes, test_maes, c=hidden_dims, cmap="viridis",
                    s=15, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="hidden_dim")
    ax.set_xlabel("Mean val MAE (last 10 epochs)")
    ax.set_ylabel("Mean test MAE (last 10 epochs)")
    ax.set_title("Val vs Test MAE")
    ax.plot([val_maes.min(), val_maes.max()],
            [val_maes.min(), val_maes.max()],
            "k--", alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.3)

    # 2. Test MAE vs num_layers, coloured by hidden_dim
    ax = axes[0][1]
    sc = ax.scatter(num_layers + np.random.default_rng(0).uniform(-0.15, 0.15, len(num_layers)),
                    test_maes, c=hidden_dims, cmap="viridis",
                    s=15, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="hidden_dim")
    ax.set_xlabel("num_layers (jittered)")
    ax.set_ylabel("Mean test MAE")
    ax.set_title("Test MAE vs depth")
    ax.grid(True, alpha=0.3)

    # 3. Test MAE vs training time
    ax = axes[1][0]
    sc = ax.scatter(times, test_maes, c=hidden_dims, cmap="viridis",
                    s=15, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="hidden_dim")
    ax.set_xlabel("Total training time (s)")
    ax.set_ylabel("Mean test MAE")
    ax.set_title("Accuracy vs runtime")
    ax.grid(True, alpha=0.3)

    # 4. Top 10 configs bar chart
    ax = axes[1][1]
    top10 = results[:10]
    labels = [
        f"hd={r['config']['hidden_dim']} nl={r['config']['num_layers']}\n"
        f"lr={r['config']['lr']:.0e} bs={r['config']['batch_size']}"
        for r in top10
    ]
    y_pos = np.arange(len(top10))
    bars = ax.barh(y_pos, [r["mean_test_mae"] for r in top10], color="C1", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean test MAE (last 10 epochs)")
    ax.set_title("Top 10 configs by val MAE")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    best = results[0]
    bc = best["config"]
    footer = (
        f"Best config (by val MAE): "
        f"hidden_dim={bc['hidden_dim']}  num_layers={bc['num_layers']}  "
        f"dropout={bc['dropout']}  lr={bc['lr']}  weight_decay={bc['weight_decay']}  "
        f"batch_size={bc['batch_size']}  activation={bc['activation']}  "
        f"batch_norm={bc['use_batch_norm']}\n"
        f"val MAE = {best['mean_val_mae']:.5f}  |  "
        f"test MAE = {best['mean_test_mae']:.5f}  |  "
        f"params = {best['num_parameters']:,}  |  "
        f"time = {best['total_train_time']:.0f}s"
    )

    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, family="monospace")
    fig.suptitle(f"GatedConv hyperparameter sweep ({len(results)} configs)", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
