"""Analyse and plot results from the GatedConv hyperparameter sweep (v2).

Only lr and weight_decay are swept; all other hyperparameters are fixed.
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
    return [r for r in payload["results"] if "mean_val_mae" in r], payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GatedConv sweep results.")
    parser.add_argument("--input", type=str, default="sweep_gated_cycle34_v2.json")
    parser.add_argument("--output", type=str, default="sweep_gated_cycle34_v2_plot.png")
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    results, payload = _load(args.input)
    if not results:
        raise SystemExit("No valid results found.")

    results.sort(key=lambda r: r["mean_val_mae"])
    print(f"Loaded {len(results)} valid runs\n")

    fixed = payload.get("fixed", {})
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
    print(f"Fixed: {fixed_str}\n")

    print(f"{'Rank':>4}  {'Val MAE':>9}  {'Test MAE':>9}  "
          f"{'lr':>9}  {'wd':>9}  {'Time':>6}")
    print("-" * 55)
    for rank, r in enumerate(results[: args.top_n], 1):
        c = r["config"]
        print(
            f"{rank:4d}  {r['mean_val_mae']:9.5f}  {r['mean_test_mae']:9.5f}  "
            f"{c['lr']:9.1e}  {c['weight_decay']:9.1e}  {r['total_train_time']:5.0f}s"
        )

    lrs = np.array([r["config"]["lr"] for r in results])
    wds = np.array([r["config"]["weight_decay"] for r in results])
    val_maes = np.array([r["mean_val_mae"] for r in results])
    test_maes = np.array([r["mean_test_mae"] for r in results])

    unique_lrs = sorted(set(lrs))
    unique_wds = sorted(set(wds))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Heatmap: test MAE by lr x weight_decay
    ax = axes[0]
    grid = np.full((len(unique_wds), len(unique_lrs)), np.nan)
    for r in results:
        lr_idx = unique_lrs.index(r["config"]["lr"])
        wd_idx = unique_wds.index(r["config"]["weight_decay"])
        grid[wd_idx, lr_idx] = r["mean_test_mae"]

    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean test MAE")
    ax.set_xticks(range(len(unique_lrs)))
    ax.set_xticklabels([f"{x:.0e}" for x in unique_lrs], rotation=45, fontsize=8)
    ax.set_yticks(range(len(unique_wds)))
    ax.set_yticklabels([f"{x:.0e}" for x in unique_wds], fontsize=8)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Weight decay")
    ax.set_title("Test MAE heatmap")

    for i in range(len(unique_wds)):
        for j in range(len(unique_lrs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center",
                        fontsize=7, color="white" if grid[i, j] > np.nanmedian(grid) else "black")

    # 2. Bar chart of all configs sorted by test MAE
    ax = axes[1]
    sorted_by_test = sorted(results, key=lambda r: r["mean_test_mae"])
    bar_labels = [f"lr={r['config']['lr']:.0e}\nwd={r['config']['weight_decay']:.0e}"
                  for r in sorted_by_test]
    y_pos = np.arange(len(sorted_by_test))
    colors = ["C1" if r == results[0] else "C0" for r in sorted_by_test]
    ax.barh(y_pos, [r["mean_test_mae"] for r in sorted_by_test],
            color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=6)
    ax.set_xlabel("Mean test MAE (last 10 epochs)")
    ax.set_title("All configs ranked by test MAE")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    best = results[0]
    bc = best["config"]
    footer = (
        f"Best (by val MAE): lr={bc['lr']}  weight_decay={bc['weight_decay']}  |  "
        f"val MAE = {best['mean_val_mae']:.5f}  |  test MAE = {best['mean_test_mae']:.5f}  |  "
        f"time = {best['total_train_time']:.0f}s\n"
        f"Fixed: {fixed_str}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, family="monospace")
    fig.suptitle(f"GatedConv lr x weight_decay sweep ({len(results)} configs)", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
