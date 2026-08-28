"""Plot atom MAE and median R_tot curves from cgcnn_extensive_comparison_curves.json."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cgcnn_extensive_comparison_curves.json")
    parser.add_argument("--output", default="cgcnn_extensive_comparison_curves.png")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    runs = payload["runs"]
    domains = ("point", "planar")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for row, domain in enumerate(domains):
        atom_key = f"{domain}_atom_only"
        tot_key = f"{domain}_atom_plus_total"
        for col, metric_key, ylabel in (
            (0, "mae", f"{payload.get('metric', 'MAE').upper()} (eV/atom)"),
            (1, "r_tot_median", r"$R_{\mathrm{tot}}$ median (%)"),
        ):
            ax = axes[row, col]
            curve_suffix = f"val_{metric_key}" if metric_key == "mae" else "val_r_tot_median"
            test_suffix = f"test_{metric_key}" if metric_key == "mae" else "test_r_tot_median"
            if atom_key in runs:
                ax.plot(
                    runs[atom_key][curve_suffix],
                    label="atom-only (val)",
                    color="C0",
                )
                ax.plot(
                    runs[atom_key][test_suffix],
                    label="atom-only (test)",
                    color="C0",
                    linestyle="--",
                    alpha=0.7,
                )
            if tot_key in runs:
                ax.plot(
                    runs[tot_key][curve_suffix],
                    label="atom+total (val)",
                    color="C1",
                )
                ax.plot(
                    runs[tot_key][test_suffix],
                    label="atom+total (test)",
                    color="C1",
                    linestyle="--",
                    alpha=0.7,
                )
            ax.set_ylabel(ylabel)
            ax.set_title(f"{domain} — {ylabel}")
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "CGCNN: per-atom MAE vs net system error (70/15/15 split)",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
