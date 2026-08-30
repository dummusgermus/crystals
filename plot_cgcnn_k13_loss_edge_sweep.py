"""Plot k=13 graph vs full-cell loss comparison (edge_k=3)."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cgcnn_k13_loss_edge_sweep_curves.json")
    parser.add_argument("--output-prefix", default="cgcnn_k13_loss_edge_sweep")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = payload.get("summary", {}).get("runs", [])
    if not rows:
        print("No runs in summary.")
        return

    modes = ["graph", "full"]
    labels = ["graph loss", "full-cell loss"]
    colors = ["C0", "C1"]

    def _value(mode: str, key: str) -> float:
        match = [r for r in rows if r["loss_mode"] == mode]
        return float(match[0][key]) if match else float("nan")

    metrics = [
        ("final_test_mae", "Test MAE (eV/atom)", "Per-atom error"),
        (
            "final_test_r_tot_graph_median",
            r"Test $R_{\mathrm{tot}}$ median (%)",
            "Graph-target net energy",
        ),
        (
            "final_test_r_tot_full_median",
            r"Test $R_{\mathrm{tot}}$ median (%)",
            "Full-cell net energy",
        ),
    ]

    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (key, ylabel, title) in zip(axes, metrics):
        vals = [_value(mode, key) for mode in modes]
        ax.bar(x, vals, color=colors, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"CGCNN k=13, edge_k=3 — graph vs full-cell loss "
        f"(λ={payload.get('lambda_tot', '?')}, {payload.get('epochs', '?')} ep)",
        y=1.02,
    )
    fig.tight_layout()
    out = f"{args.output_prefix}_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
