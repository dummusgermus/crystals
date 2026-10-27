#!/usr/bin/env python3
"""Compare prior predictions_new (k=8, graph global loss) vs current (k=13, optimized).

Prior delivery stats are loaded from ``prediction_old_vs_new_comparison.json``
(blocks["new"], archived before the k13 restore overwrote CSVs). Current stats
are recomputed from ``predictions_new/``.

Usage:
    python analyze_predictions_k8_vs_k13.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from analyze_predictions_old_vs_new import (
    DeliveryStats,
    block_stats,
    fmt,
    fmt_pct,
    latex_escape_texttt,
    load_delivery,
    top_outliers,
)
from analyze_total_error_vs_delta import CrystalStats, iter_csvs, load_crystal

ROOT = Path(__file__).resolve().parent
PRIOR_JSON = ROOT / "prediction_old_vs_new_comparison.json"
NEW_ROOT = ROOT / "predictions_new"
CURVES_JSON = ROOT / "delivery_global_v2_curves.json"
DEFAULT_JSON = ROOT / "predictions_k8_vs_k13_comparison.json"
DEFAULT_TEX = ROOT / "predictions_k8_vs_k13_comparison.tex"
MIN_DELTA = 1e-6


def load_prior_blocks(path: Path) -> Dict[str, Dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["blocks"]["new"]


def load_prior_train(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("training_val_checkpoint", {})


def load_current_train(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, r in payload.get("runs", {}).items():
        out[k] = {
            "lambda_tot": r.get("lambda_tot"),
            "best_val_r_tot_median": r.get("best_val_r_tot_median"),
            "best_val_mae": r.get("best_val_mae"),
            "total_target_mode": r.get("total_target_mode"),
            "final_val_r_tot_graph_median": r.get("final_val_r_tot_graph_median"),
            "final_val_r_tot_full_median": r.get("final_val_r_tot_full_median"),
        }
    return out


def compute_blocks(rows: List[CrystalStats], label: str) -> Dict[str, Dict[str, dict]]:
    blocks: Dict[str, Dict[str, dict]] = {}
    for domain in ("point", "planar"):
        blocks[domain] = {}
        for split in ("all", "validation"):
            subset = [r for r in rows if r.domain == domain]
            if split == "validation":
                subset = [r for r in subset if r.split == "validation"]
            blocks[domain][split] = asdict(
                block_stats(label, subset, domain=domain, split=split)
            )
    return blocks


def delta_pp(new_val: float, old_val: float) -> float:
    return new_val - old_val


def improved_lower_is_better(new_val: float, old_val: float) -> bool:
    return new_val < old_val - 1e-12


def write_latex(payload: dict, path: Path) -> None:
    prior = payload["blocks"]["prior"]
    current = payload["blocks"]["current"]
    train_p = payload["training"]["prior"]
    train_c = payload["training"]["current"]

    def row(domain: str, split: str, key: str, *, nested: Optional[str] = None):
        o = prior[domain][split]
        n = current[domain][split]
        if nested:
            return o[nested][key], n[nested][key]
        return o[key], n[key]

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\usepackage{parskip}",
        r"\geometry{margin=2cm}",
        r"",
        r"\begin{document}",
        r"",
        r"\title{Delivery update: \texttt{predictions\_new} before vs.\ after k=13 optimization}",
        r"\date{}",
        r"\maketitle",
        r"",
        r"\noindent\textbf{Compared deliveries.}",
        r"\begin{itemize}",
        r"  \item \textbf{Prior} (Aug~2026): $k=8$ point graphs; global-loss v2 with "
        r"\texttt{graph} net-energy target; CGCNN/Transformer point $\lambda=0.01$; "
        r"planar Transformer $\lambda=0.005$.",
        r"  \item \textbf{Current} (Aug~2026): $k=13$, edge\_k$=3$ point graphs; "
        r"point CGCNN full-cell global loss ($\lambda=0.01$); point Transformer "
        r"$\lambda=0.02$; planar Transformer $\lambda=0.01$; tuned CGCNN/planar $\lambda$ unchanged.",
        r"\end{itemize}",
        r"Same crystals and 90/10 split (seed~42). Prior CSV aggregates are archived in "
        r"\texttt{prediction\_old\_vs\_new\_comparison.json} (``blocks.new''); current stats "
        r"were recomputed after restore.",
        r"",
        r"\section*{Training checkpoint metrics (model selection)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{l l r r r r}",
        r"  \toprule",
        r"  Model & Target & $\lambda$ & Prior val $R_{\mathrm{tot}}$ & Current val $R_{\mathrm{tot}}$ & Current val MAE \\",
        r"  \midrule",
    ]
    pretty = [
        ("point_cgcnn", "Point CGCNN"),
        ("point_transformer", "Point Transformer"),
        ("planar_cgcnn", "Planar CGCNN"),
        ("planar_transformer", "Planar Transformer"),
    ]
    for key, name in pretty:
        tp = train_p.get(key, {})
        tc = train_c.get(key, {})
        target = tc.get("total_target_mode") or "graph"
        lines.append(
            f"  {name} & {target} & {tc.get('lambda_tot', tp.get('lambda_tot', 0)):g} & "
            f"{tp.get('best_val_r_tot_median', float('nan')):.1f} & "
            f"{tc.get('best_val_r_tot_median', float('nan')):.1f} & "
            f"{tc.get('best_val_mae', float('nan')):.4f} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\noindent\textit{Note:} Prior point CGCNN checkpoint used graph-target $R_{\mathrm{tot}}$; "
        r"current point CGCNN uses full-cell $R_{\mathrm{tot}}$ for selection "
        f"(graph-target val $R_{{\\mathrm{{tot}}}}$ = "
        f"{train_c.get('point_cgcnn', {}).get('final_val_r_tot_graph_median', float('nan')):.1f}\\%).",
        r"",
        r"\section*{Table 1 | Validation per-atom MAE (eV/atom, mean)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{l r r r r}",
        r"  \toprule",
        r"  Domain & Model & Prior & Current & $\Delta$ \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for model, label in [("mae_cgcnn_mean", "CGCNN"), ("mae_transformer_mean", "Transformer")]:
            ov, nv = row(domain, "validation", model)
            d = delta_pp(nv, ov)
            sign = "better" if improved_lower_is_better(nv, ov) else "worse"
            lines.append(
                f"  {domain.capitalize():6s} & {label:11s} & {fmt(ov)} & {fmt(nv)} & "
                f"{d:+.4f} ({sign}) \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 2 | Validation $R_{\mathrm{tot}}$ median (\%)}",
        r"",
        r"Crystals with $|\Delta E_{\mathrm{true}}|\ge 10^{-6}$~eV. Primary net-energy metric.",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{l r r r r r}",
        r"  \toprule",
        r"  Domain & Model & Prior & Current & $\Delta$ (pp) & Verdict \\",
        r"  \midrule",
    ]
    verdict_rows = []
    for domain in ("point", "planar"):
        for nested, label in [("rtot_cgcnn", "CGCNN"), ("rtot_transformer", "Transformer")]:
            ov, nv = row(domain, "validation", "median", nested=nested)
            d = delta_pp(nv, ov)
            if abs(d) < 0.05:
                verdict = "$\approx$ flat"
            elif improved_lower_is_better(nv, ov):
                verdict = "improved"
            else:
                verdict = "worse"
            verdict_rows.append((domain, label, d, verdict))
            lines.append(
                f"  {domain.capitalize():6s} & {label:11s} & {fmt(ov, 2)} & {fmt(nv, 2)} & "
                f"{d:+.2f} & {verdict} \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 3 | Validation absolute total error $|\mathrm{PE}_{\mathrm{pred}}^{\mathrm{tot}}-\mathrm{PE}_{\mathrm{true}}^{\mathrm{tot}}|$ (eV)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{l r r r r r}",
        r"  \toprule",
        r"  Domain & Model & Prior med. & Current med. & Prior mean & Current mean \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        oc_m, nc_m = row(domain, "validation", "abs_err_cgcnn_median")
        oc_a, nc_a = row(domain, "validation", "abs_err_cgcnn_mean")
        ot_m, nt_m = row(domain, "validation", "abs_err_transformer_median")
        ot_a, nt_a = row(domain, "validation", "abs_err_transformer_mean")
        lines.append(
            f"  {domain.capitalize():6s} & CGCNN & {fmt(oc_m)} & {fmt(nc_m)} & "
            f"{fmt(oc_a, 2)} & {fmt(nc_a, 2)} \\\\"
        )
        lines.append(
            f"  & Transformer & {fmt(ot_m)} & {fmt(nt_m)} & "
            f"{fmt(ot_a, 2)} & {fmt(nt_a, 2)} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 4 | All crystals ($R_{\mathrm{tot}}$ median, \%)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{l r r r r}",
        r"  \toprule",
        r"  Domain & Model & Prior & Current & $\Delta$ (pp) \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for nested, label in [("rtot_cgcnn", "CGCNN"), ("rtot_transformer", "Transformer")]:
            ov, nv = row(domain, "all", "median", nested=nested)
            lines.append(
                f"  {domain.capitalize():6s} & {label:11s} & {fmt(ov, 2)} & {fmt(nv, 2)} & "
                f"{delta_pp(nv, ov):+.2f} \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Assessment}",
        r"\begin{itemize}",
    ]

    # Build assessment bullets from validation R_tot
    for domain, label, d, verdict in verdict_rows:
        ov, nv = row(domain, "validation", "median", nested=f"rtot_{label.lower()}")
        if domain == "point" and label == "CGCNN":
            lines.append(
                r"  \item \textbf{Point CGCNN (primary goal):} validation $R_{\mathrm{tot}}$ "
                f"median {fmt(ov, 1)}\\% $\\rightarrow$ {fmt(nv, 1)}\\% "
                f"({d:+.1f}~pp). "
                + (
                    r"Net-energy generalization \textbf{did not improve} on held-out validation despite "
                    r"$k=13$ and full-cell training loss; checkpoint metric changed (graph $\\rightarrow$ full)."
                    if not improved_lower_is_better(nv, ov)
                    else r"Net-energy generalization \textbf{improved}."
                )
            )
        elif domain == "point" and label == "Transformer":
            lines.append(
                r"  \item \textbf{Point Transformer:} validation $R_{\mathrm{tot}}$ "
                f"{fmt(ov, 1)}\\% $\\rightarrow$ {fmt(nv, 1)}\\% ({d:+.1f}~pp); "
                f"per-atom MAE improved ({fmt(row('point','validation','mae_transformer_mean')[0])} "
                f"$\\rightarrow$ {fmt(row('point','validation','mae_transformer_mean')[1])}~eV)."
            )
        elif domain == "planar" and label == "CGCNN":
            lines.append(
                r"  \item \textbf{Planar CGCNN:} validation $R_{\mathrm{tot}}$ "
                f"{fmt(ov, 1)}\\% $\\rightarrow$ {fmt(nv, 1)}\\% ({d:+.1f}~pp)."
            )
        elif domain == "planar" and label == "Transformer":
            lines.append(
                r"  \item \textbf{Planar Transformer:} validation $R_{\mathrm{tot}}$ "
                f"{fmt(ov, 1)}\\% $\\rightarrow$ {fmt(nv, 1)}\\% ({d:+.1f}~pp); "
                r"$\lambda$ increased $0.005\\rightarrow 0.01$ per sweep."
            )

    pc_prior, pc_curr = row("point", "all", "median", nested="rtot_cgcnn")
    lines.append(
        r"  \item \textbf{Point all crystals (train+val):} CGCNN $R_{\mathrm{tot}}$ median "
        f"{fmt(pc_prior, 1)}\\% $\\rightarrow$ {fmt(pc_curr, 1)}\\% "
        f"({delta_pp(pc_curr, pc_prior):+.1f}~pp) --- modest shift on full delivery set."
    )
    lines.append(
        r"  \item \textbf{Caveat:} Point cells have many configs with $|\Delta E_{\mathrm{true}}|\approx 0$; "
        r"absolute eV errors can look large while $R_{\mathrm{tot}}$ is the fair net-energy metric. "
        r"Training val $R_{\mathrm{tot}}$ for point CGCNN rose because checkpoint target switched to full-cell."
    )
    lines.append(
        r"  \item \textbf{Bottom line:} Mixed results. Point Transformer and planar metrics show "
        r"small improvements or flat validation $R_{\mathrm{tot}}$; point CGCNN validation $R_{\mathrm{tot}}$ "
        r"did not beat the prior $k=8$ graph-loss delivery on this split. Per-atom MAE remains excellent "
        r"($\sim 0.003$~eV on point validation)."
    )
    lines += [
        r"\end{itemize}",
        r"",
        r"\section*{Largest point validation outliers (current CGCNN)}",
        r"",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{l r r r r}",
        r"  \toprule",
        r"  Config & $\Delta E_{\mathrm{true}}$ & MAE & $|\mathrm{err}|$ & $R_{\mathrm{tot}}$ \\",
        r"  \midrule",
    ]
    for row_o in payload["outliers"]["current"]["point"]["validation"][:8]:
        label = latex_escape_texttt(f"{row_o['folder']}/{row_o['config']}")
        lines.append(
            f"  \\texttt{{{label}}} & {fmt(row_o['delta_true_eV'])} & "
            f"{fmt(row_o['mae_cgcnn_eV'])} & {fmt(row_o['abs_err_cgcnn_eV'])} & "
            f"{fmt(row_o['rtot_cgcnn_pct'], 1)} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\end{document}",
        r"",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prior vs k13 predictions_new.")
    parser.add_argument("--prior-json", type=Path, default=PRIOR_JSON)
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--curves-json", type=Path, default=CURVES_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--latex-out", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()

    print("Loading prior stats from archive …", flush=True)
    prior_blocks = load_prior_blocks(args.prior_json)
    prior_train = load_prior_train(args.prior_json)

    print("Loading current predictions_new …", flush=True)
    rows = load_delivery(args.new_root)
    current_blocks = compute_blocks(rows, "current")
    current_train = load_current_train(args.curves_json)

    outliers = {
        "current": {
            "point": {
                "validation": top_outliers(rows, domain="point", split="validation")
            }
        }
    }

    payload = {
        "prior_source": str(args.prior_json),
        "current_root": str(args.new_root),
        "blocks": {"prior": prior_blocks, "current": current_blocks},
        "training": {"prior": prior_train, "current": current_train},
        "outliers": outliers,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_latex(payload, args.latex_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.latex_out}")

    pv = prior_blocks["point"]["validation"]
    cv = current_blocks["point"]["validation"]
    print("\nPoint validation summary:")
    print(
        f"  R_tot CGCNN med:  {pv['rtot_cgcnn']['median']:.1f}% -> "
        f"{cv['rtot_cgcnn']['median']:.1f}%"
    )
    print(
        f"  R_tot Trans med:  {pv['rtot_transformer']['median']:.1f}% -> "
        f"{cv['rtot_transformer']['median']:.1f}%"
    )
    print(
        f"  MAE CGCNN:        {pv['mae_cgcnn_mean']:.4f} -> {cv['mae_cgcnn_mean']:.4f} eV"
    )


if __name__ == "__main__":
    main()
