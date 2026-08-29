"""Plot CGCNN graph-fraction sweep: MAE and R_tot vs graph size."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cgcnn_graph_fraction_sweep_curves.json")
    parser.add_argument(
        "--output-prefix",
        default="cgcnn_graph_fraction_sweep",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    tiers = payload.get("summary", {}).get("tiers", [])
    if not tiers:
        print("No tier summary found.")
        return

    x = [100.0 * float(t.get("subset_fraction_actual_median", 0.0)) for t in tiers]
    mae = [float(t["final_test_mae"]) for t in tiers]
    r_graph = [float(t["final_test_r_tot_median"]) for t in tiers]
    r_full = [float(t.get("final_test_r_tot_full_median", np.nan)) for t in tiers]
    labels = [str(t["tag"]) for t in tiers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(x, mae, "o-", color="C0", linewidth=2, label="Test MAE")
    ax.set_xlabel("Graph atoms (% of full cell, median)")
    ax.set_ylabel("Test MAE (eV/atom)")
    ax.set_title("Per-atom error vs graph size")
    ax.grid(True, alpha=0.25)
    for xi, yi, lab in zip(x, mae, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax = axes[1]
    ax.plot(x, r_graph, "o-", color="C1", linewidth=2, label=r"$R_{\mathrm{tot}}$ graph")
    ax.plot(x, r_full, "s--", color="C3", linewidth=2, label=r"$R_{\mathrm{tot}}$ full cell")
    ax.set_xlabel("Graph atoms (% of full cell, median)")
    ax.set_ylabel(r"Test $R_{\mathrm{tot}}$ median (%)")
    ax.set_title("Net-energy error vs graph size")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.suptitle("CGCNN global v2 — point graph fraction sweep", y=1.02)
    fig.tight_layout()
    out = f"{args.output_prefix}_metrics.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
