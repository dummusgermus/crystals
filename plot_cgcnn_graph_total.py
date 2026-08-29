"""Plot graph-total CGCNN training curves and test-set pred vs true."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _plot_epoch_curves(payload: dict, out_path: str) -> None:
    runs = payload["runs"]
    domains = [d for d in ("point", "planar") if d in runs]
    panels = [
        ("val_mae_eV", "Val MAE (eV)"),
        ("val_r_tot_median", "Val $R_{\\mathrm{tot}}$ median (%)"),
        ("test_r_tot_median", "Test $R_{\\mathrm{tot}}$ median (%)"),
        ("test_abs_err_eV", "Test mean |error| (eV)"),
    ]
    nrows = len(panels)
    ncols = len(domains)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows), sharex="col")
    if ncols == 1:
        axes = np.array([[ax] for ax in axes])
    if nrows == 1:
        axes = np.array([axes])

    for col, domain in enumerate(domains):
        run = runs[domain]
        for row, (key, ylabel) in enumerate(panels):
            ax = axes[row, col]
            if key in run:
                ax.plot(run[key], linewidth=1.8)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(domain.capitalize())
            if row == nrows - 1:
                ax.set_xlabel("Epoch")
    fig.suptitle("Graph-level total residual CGCNN", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_scatter_and_rtot(payload: dict, out_path: str) -> None:
    runs = payload["runs"]
    domains = [d for d in ("point", "planar") if d in runs]
    fig, axes = plt.subplots(1, len(domains) * 2, figsize=(6 * len(domains), 5))
    if len(domains) * 2 == 1:
        axes = [axes]
    axes = np.atleast_1d(axes).ravel()

    for i, domain in enumerate(domains):
        preds = runs[domain].get("test_predictions", [])
        if not preds:
            continue
        true = np.array([p["true_eV"] for p in preds])
        pred = np.array([p["pred_eV"] for p in preds])
        rtot = np.array([p["r_tot_pct"] for p in preds if p["r_tot_pct"] == p["r_tot_pct"]])

        ax_sc = axes[2 * i]
        lim = max(np.max(np.abs(true)), np.max(np.abs(pred)), 1e-3)
        ax_sc.scatter(true, pred, s=12, alpha=0.6, edgecolors="none")
        ax_sc.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5)
        ax_sc.set_xlabel(r"True $\Delta E_{\mathrm{tot}}$ (eV)")
        ax_sc.set_ylabel(r"Predicted $\Delta E_{\mathrm{tot}}$ (eV)")
        ax_sc.set_title(f"{domain}: test pred vs true")
        ax_sc.grid(True, alpha=0.25)

        ax_hist = axes[2 * i + 1]
        if len(rtot):
            ax_hist.hist(rtot, bins=30, color="steelblue", alpha=0.85, edgecolor="white")
            med = float(np.median(rtot))
            ax_hist.axvline(med, color="crimson", linestyle="--", label=f"median={med:.1f}%")
            ax_hist.axvline(100, color="black", linestyle=":", alpha=0.6, label="100% baseline")
            ax_hist.legend(fontsize=8)
        ax_hist.set_xlabel(r"$R_{\mathrm{tot}}$ (%)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"{domain}: test $R_{{\\mathrm{{tot}}}}$")
        ax_hist.grid(True, alpha=0.25)

    fig.suptitle("Direct graph-level total residual prediction (test set)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot graph-total CGCNN results.")
    parser.add_argument("--input", default=os.path.join(ROOT, "cgcnn_graph_total_curves.json"))
    parser.add_argument(
        "--output-prefix",
        default=os.path.join(ROOT, "cgcnn_graph_total"),
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    _plot_epoch_curves(payload, f"{args.output_prefix}_epoch_curves.png")
    _plot_scatter_and_rtot(payload, f"{args.output_prefix}_test_scatter.png")


if __name__ == "__main__":
    main()
