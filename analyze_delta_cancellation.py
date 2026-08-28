#!/usr/bin/env python3
"""Analyze per-atom delta cancellation vs net system change."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

HEADER_RE = re.compile(r"^#\s*([^=]+)=\s*([+-]?[\d.eE+-]+)\s*$")


def load(path: Path) -> dict:
    meta: dict = {}
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# domain="):
                meta["domain"] = line.split("=", 1)[1].strip()
            elif line.startswith("atom_id,"):
                reader = csv.DictReader(fh, fieldnames=line.strip().split(","))
                rows.extend(reader)
            else:
                m = HEADER_RE.match(line.strip())
                if m:
                    meta[m.group(1).strip()] = float(m.group(2))

    init = np.array([float(r["pe_initial"]) for r in rows], dtype=np.float64)
    true = np.array([float(r["pe_true"]) for r in rows], dtype=np.float64)
    pred_c = np.array([float(r["pe_pred_cgcnn"]) for r in rows], dtype=np.float64)

    d_true = true - init
    d_pred = pred_c - init

    meta["n"] = len(rows)
    meta["delta_total"] = float(d_true.sum())
    meta["sum_abs_delta_true"] = float(np.abs(d_true).sum())
    meta["mean_abs_delta_true"] = float(np.mean(np.abs(d_true)))
    meta["max_abs_delta_true"] = float(np.max(np.abs(d_true)))
    meta["cancel_true"] = (
        1.0 - abs(meta["delta_total"]) / meta["sum_abs_delta_true"]
        if meta["sum_abs_delta_true"]
        else 0.0
    )
    meta["mae_delta_pred"] = float(np.mean(np.abs(d_pred - d_true)))
    meta["delta_pred_total"] = float(d_pred.sum())
    meta["mae"] = float(meta.get("mae_cgcnn_abs_eV", np.mean(np.abs(pred_c - true))))
    meta["ec"] = float(meta.get("pe_error_cgcnn_total_eV", float((pred_c - true).sum())))
    return meta


def bucket(arr: list[dict], lo: float, hi: float | None) -> list[dict]:
    out = []
    for m in arr:
        ad = abs(m["delta_total"])
        if hi is None:
            if ad >= lo:
                out.append(m)
        elif lo <= ad < hi:
            out.append(m)
    return out


def main() -> None:
    all_meta = [load(p) for p in sorted(Path("predictions").rglob("*.csv"))]

    ex_path = Path("predictions/planar/Ag-Be_XY'X/pe_table.csv")
    ex = load(ex_path)
    print("=== Ag-Be_XY'X ===")
    for k in (
        "n",
        "delta_total",
        "sum_abs_delta_true",
        "mean_abs_delta_true",
        "max_abs_delta_true",
        "cancel_true",
        "mae",
        "mae_delta_pred",
        "delta_pred_total",
        "ec",
    ):
        print(f"  {k}: {ex[k]:.6g}")

    bins = [
        ("|net delta| < 0.05 eV", 0.0, 0.05),
        ("0.05 – 0.2 eV", 0.05, 0.2),
        ("0.2 – 1 eV", 0.2, 1.0),
        ("> 1 eV", 1.0, None),
    ]
    print("\n=== Buckets (all deliveries) ===")
    for title, lo, hi in bins:
        arr = bucket(all_meta, lo, hi)
        if not arr:
            continue
        rel = [
            100 * abs(m["ec"]) / abs(m["delta_total"])
            for m in arr
            if abs(m["delta_total"]) > 1e-6
        ]
        print(f"\n{title}  n={len(arr)}")
        print(f"  mean |net delta|:           {np.mean([abs(m['delta_total']) for m in arr]):.4g} eV")
        print(f"  mean sum|atom delta|:       {np.mean([m['sum_abs_delta_true'] for m in arr]):.4g} eV")
        print(f"  mean per-atom |delta|:      {np.mean([m['mean_abs_delta_true'] for m in arr]):.4g} eV")
        print(f"  mean cancellation:         {np.mean([m['cancel_true'] for m in arr]):.1%}")
        print(f"  mean MAE (abs PE):         {np.mean([m['mae'] for m in arr]):.4g} eV/atom")
        print(f"  mean MAE (delta):          {np.mean([m['mae_delta_pred'] for m in arr]):.4g} eV/atom")
        if rel:
            print(f"  mean |total err|/|delta|:  {np.mean(rel):.1f}%  median {np.median(rel):.1f}%")

    planar_small = [m for m in all_meta if m.get("domain") == "planar" and abs(m["delta_total"]) < 0.05]
    print(f"\n=== Planar small-net-change (|delta|<0.05), n={len(planar_small)} ===")
    print(f"  mean cancellation: {np.mean([m['cancel_true'] for m in planar_small]):.1%}")
    print(f"  mean per-atom |delta|: {np.mean([m['mean_abs_delta_true'] for m in planar_small]):.4g} eV")
    print(f"  mean MAE on delta: {np.mean([m['mae_delta_pred'] for m in planar_small]):.4g} eV")


if __name__ == "__main__":
    main()
