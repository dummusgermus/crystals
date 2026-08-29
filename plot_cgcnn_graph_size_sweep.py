"""Plot CGCNN graph shell-size sweep: metrics + timing scaling."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cgcnn_graph_size_sweep_curves.json")
    parser.add_argument("--output-prefix", default="cgcnn_graph_size_sweep")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    tiers = payload.get("summary", {}).get("tiers", [])
    if not tiers:
        print("No tiers in summary.")
        return

    k = [t["cutoff_k"] for t in tiers]
    nodes = [t.get("subset_size_median") or np.nan for t in tiers]
    mae = [t["final_test_mae"] for t in tiers]
    r_g = [t["final_test_r_tot_median"] for t in tiers]
    r_f = [t.get("final_test_r_tot_full_median", np.nan) for t in tiers]
    train_min = [t.get("train_wall_s", np.nan) / 60.0 for t in tiers]
    infer_ms = [t.get("inference_ms_per_graph", np.nan) for t in tiers]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(k, mae, "o-", color="C0")
    ax.set_xlabel("cutoff_k")
    ax.set_ylabel("Test MAE (eV/atom)")
    ax.set_title("Per-atom error")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(k, r_g, "o-", label="graph target")
    ax.plot(k, r_f, "s--", label="full cell")
    ax.set_xlabel("cutoff_k")
    ax.set_ylabel(r"Test $R_{\mathrm{tot}}$ median (%)")
    ax.legend(fontsize=8)
    ax.set_title("Net-energy error")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(k, train_min, "o-", color="C2")
    ax.set_xlabel("cutoff_k")
    ax.set_ylabel("Training time (min)")
    ax.set_title(f"Training ({payload.get('epochs', '?')} epochs)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(nodes, infer_ms, "o-", color="C4")
    ax.set_xlabel("Median nodes / graph")
    ax.set_ylabel("Inference (ms / test graph)")
    ax.set_title("Test-set forward pass")
    ax.grid(True, alpha=0.25)

    fig.suptitle("CGCNN global v2 — graph shell size sweep", y=1.01)
    fig.tight_layout()
    out = f"{args.output_prefix}_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
