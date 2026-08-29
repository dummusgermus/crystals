#!/usr/bin/env python3
"""Compare original predictions/ vs global-v2 predictions_new/ deliveries.

Reads merged CSV headers (same layout as analyze_total_error_vs_delta.py) and
writes a JSON summary plus a LaTeX report.

Usage:
    python analyze_predictions_old_vs_new.py
    python analyze_predictions_old_vs_new.py --latex-out prediction_old_vs_new_comparison.tex
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from analyze_total_error_vs_delta import CrystalStats, iter_csvs, load_crystal, summarize


ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT / "predictions"
NEW_ROOT = ROOT / "predictions_new"
DEFAULT_JSON = ROOT / "prediction_old_vs_new_comparison.json"
DEFAULT_TEX = ROOT / "prediction_old_vs_new_comparison.tex"
MIN_DELTA = 1e-6


@dataclass
class DeliveryStats:
    label: str
    domain: str
    split: str
    n_csv: int
    n_rtot: int
    mae_cgcnn_mean: float
    mae_transformer_mean: float
    abs_err_cgcnn_mean: float
    abs_err_cgcnn_median: float
    abs_err_transformer_mean: float
    abs_err_transformer_median: float
    rtot_cgcnn: Dict[str, float]
    rtot_transformer: Dict[str, float]
    frac_rtot_c_below_100: float
    frac_rtot_c_below_50: float
    frac_rtot_t_below_100: float
    frac_rtot_t_below_50: float


def load_delivery(root: Path) -> List[CrystalStats]:
    rows: List[CrystalStats] = []
    for path in iter_csvs(root):
        row = load_crystal(path)
        if row is not None:
            rows.append(row)
    return rows


def crystal_id(row: CrystalStats) -> Tuple[str, str, str]:
    parts = Path(row.path).parts
    folder = parts[-2] if len(parts) >= 2 else ""
    return (row.domain, folder, row.config)


def block_stats(
    label: str,
    rows: List[CrystalStats],
    *,
    domain: str,
    split: str,
) -> DeliveryStats:
    rtot_rows = [r for r in rows if abs(r.delta_true_eV) >= MIN_DELTA]
    rc = [r.rel_cgcnn_pct for r in rtot_rows]
    rt = [r.rel_transformer_pct for r in rtot_rows]
    sc = summarize(rc)
    st = summarize(rt)
    abs_c = [abs(r.pe_error_cgcnn_eV) for r in rows]
    abs_t = [abs(r.pe_error_transformer_eV) for r in rows]
    return DeliveryStats(
        label=label,
        domain=domain,
        split=split,
        n_csv=len(rows),
        n_rtot=len(rtot_rows),
        mae_cgcnn_mean=statistics.fmean([r.mae_cgcnn_eV for r in rows]) if rows else 0.0,
        mae_transformer_mean=statistics.fmean([r.mae_transformer_eV for r in rows])
        if rows
        else 0.0,
        abs_err_cgcnn_mean=statistics.fmean(abs_c) if abs_c else 0.0,
        abs_err_cgcnn_median=statistics.median(abs_c) if abs_c else 0.0,
        abs_err_transformer_mean=statistics.fmean(abs_t) if abs_t else 0.0,
        abs_err_transformer_median=statistics.median(abs_t) if abs_t else 0.0,
        rtot_cgcnn=sc,
        rtot_transformer=st,
        frac_rtot_c_below_100=sc.get("frac_below_100pct", 0.0),
        frac_rtot_c_below_50=sc.get("frac_below_50pct", 0.0),
        frac_rtot_t_below_100=st.get("frac_below_100pct", 0.0),
        frac_rtot_t_below_50=st.get("frac_below_50pct", 0.0),
    )


def paired_improvements(
    old_rows: List[CrystalStats],
    new_rows: List[CrystalStats],
    *,
    domain: str,
    split: str,
    metric: str,
) -> Dict[str, float]:
    old_map = {crystal_id(r): r for r in old_rows if r.domain == domain and r.split == split}
    new_map = {crystal_id(r): r for r in new_rows if r.domain == domain and r.split == split}
    keys = sorted(set(old_map) & set(new_map))
    improved = 0
    worsened = 0
    ties = 0
    deltas: List[float] = []
    for k in keys:
        o, n = old_map[k], new_map[k]
        if abs(o.delta_true_eV) < MIN_DELTA:
            continue
        if metric == "rtot_cgcnn":
            ov, nv = o.rel_cgcnn_pct, n.rel_cgcnn_pct
        elif metric == "rtot_transformer":
            ov, nv = o.rel_transformer_pct, n.rel_transformer_pct
        elif metric == "mae_cgcnn":
            ov, nv = o.mae_cgcnn_eV, n.mae_cgcnn_eV
        elif metric == "mae_transformer":
            ov, nv = o.mae_transformer_eV, n.mae_transformer_eV
        elif metric == "abs_err_cgcnn":
            ov, nv = abs(o.pe_error_cgcnn_eV), abs(n.pe_error_cgcnn_eV)
        elif metric == "abs_err_transformer":
            ov, nv = abs(o.pe_error_transformer_eV), abs(n.pe_error_transformer_eV)
        else:
            raise ValueError(metric)
        deltas.append(nv - ov)
        if nv < ov - 1e-12:
            improved += 1
        elif nv > ov + 1e-12:
            worsened += 1
        else:
            ties += 1
    n = len(deltas)
    return {
        "n": n,
        "improved": improved,
        "worsened": worsened,
        "ties": ties,
        "pct_improved": 100.0 * improved / n if n else 0.0,
        "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
        "median_delta": statistics.median(deltas) if deltas else 0.0,
    }


def delta_bucket_stats(rows: List[CrystalStats], *, domain: str, split: str) -> List[Dict]:
    subset = [r for r in rows if r.domain == domain and r.split == split]
    buckets = [
        ("$|\\Delta E|<10^{-6}$ eV (net $\\approx 0$)", lambda d: abs(d) < MIN_DELTA),
        ("$10^{-6}$--$0.05$ eV", lambda d: MIN_DELTA <= abs(d) < 0.05),
        ("$0.05$--$0.5$ eV", lambda d: 0.05 <= abs(d) < 0.5),
        ("$\\ge 0.5$ eV", lambda d: abs(d) >= 0.5),
    ]
    out = []
    for name, fn in buckets:
        b = [r for r in subset if fn(r.delta_true_eV)]
        if not b:
            continue
        abs_c = [abs(r.pe_error_cgcnn_eV) for r in b]
        rc = [r.rel_cgcnn_pct for r in b if abs(r.delta_true_eV) >= MIN_DELTA]
        out.append(
            {
                "bucket": name,
                "n": len(b),
                "mean_delta_true_eV": statistics.fmean([abs(r.delta_true_eV) for r in b]),
                "mae_cgcnn_mean": statistics.fmean([r.mae_cgcnn_eV for r in b]),
                "abs_err_cgcnn_mean": statistics.fmean(abs_c),
                "abs_err_cgcnn_median": statistics.median(abs_c),
                "rtot_cgcnn_median": statistics.median(rc) if rc else float("nan"),
            }
        )
    return out


def top_outliers(
    rows: List[CrystalStats], *, domain: str, split: str, n: int = 8
) -> List[Dict]:
    subset = [
        r
        for r in rows
        if r.domain == domain and r.split == split and abs(r.delta_true_eV) >= MIN_DELTA
    ]
    subset.sort(key=lambda r: abs(r.pe_error_cgcnn_eV), reverse=True)
    out = []
    for r in subset[:n]:
        out.append(
            {
                "folder": Path(r.path).parts[-2],
                "config": r.config,
                "delta_true_eV": r.delta_true_eV,
                "mae_cgcnn_eV": r.mae_cgcnn_eV,
                "abs_err_cgcnn_eV": abs(r.pe_error_cgcnn_eV),
                "rtot_cgcnn_pct": r.rel_cgcnn_pct,
            }
        )
    return out


def fmt(x: float, nd: int = 4) -> str:
    if x != x:
        return "---"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{nd}f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1f}\\%"


def latex_escape_texttt(text: str) -> str:
    """Escape characters that break LaTeX inside \\texttt{...}."""
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def write_latex(payload: dict, path: Path) -> None:
    train = payload["training_val_checkpoint"]
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{geometry}",
        r"\usepackage{parskip}",
        r"\usepackage{tabularx}",
        r"\geometry{margin=2cm}",
        r"",
        r"\begin{document}",
        r"",
        r"\title{Comparison: Original Delivery vs.\ Global-Loss v2 (\texttt{predictions\_new})}",
        r"\date{}",
        r"\maketitle",
        r"",
        r"\noindent\textbf{Setup.} Same crystals, same 90/10 within-group split (seed~42), "
        r"same architectures and 2000 training epochs. Only the loss and checkpoint criterion "
        r"changed: atom MAE $\rightarrow$ global-loss v2 with best validation $R_{\mathrm{tot}}$.",
        r"",
        r"\medskip",
        r"\noindent\textbf{Important: median $|\Delta E_{\mathrm{pred}}^{\mathrm{tot}}| \approx 0.11$~eV "
        r"on point validation is \emph{not} per-atom MAE.} Per-atom MAE there is $\sim 0.003$~eV. "
        r"The 0.11~eV number is the median absolute \emph{signed net error} over full cells "
        r"($\sim$3000 atoms). Many point configs have $|\Delta E_{\mathrm{true}}|\approx 0$; "
        r"any small net bias then appears as a finite eV error even when per-atom accuracy is excellent. "
        r"Use $R_{\mathrm{tot}}$ for net-energy quality.",
        r"",
        r"\section*{Training validation metrics (checkpoint selection)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"  \toprule",
        r"  Model & $\lambda_{\mathrm{tot}}$ & val $R_{\mathrm{tot}}$ (\%) & val MAE (eV) \\",
        r"  \midrule",
    ]
    for model_key, pretty in [
        ("point_cgcnn", "Point CGCNN"),
        ("point_transformer", "Point Transformer"),
        ("planar_cgcnn", "Planar CGCNN"),
        ("planar_transformer", "Planar Transformer"),
    ]:
        t = train[model_key]
        lines.append(
            f"  {pretty} & {t['lambda_tot']:g} & {t['best_val_r_tot_median']:.1f} & "
            f"{t['best_val_mae']:.4f} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 1 | Per-atom MAE (eV/atom, mean over crystals)}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"  \toprule",
        r"  Domain & Split & Old CGCNN & New CGCNN & Old Trans. & New Trans. \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for split in ("all", "validation"):
            o = payload["blocks"]["old"][domain][split]
            n = payload["blocks"]["new"][domain][split]
            split_label = "all" if split == "all" else "validation"
            lines.append(
                f"  {domain.capitalize():6s} & {split_label:10s} & "
                f"{fmt(o['mae_cgcnn_mean'])} & {fmt(n['mae_cgcnn_mean'])} & "
                f"{fmt(o['mae_transformer_mean'])} & {fmt(n['mae_transformer_mean'])} \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 2 | Absolute total error $|\mathrm{PE}_{\mathrm{pred}}^{\mathrm{tot}}-\mathrm{PE}_{\mathrm{true}}^{\mathrm{tot}}|$ (eV)}",
        r"",
        r"Mean and median over crystals. Large point cells inflate the mean; see bucket breakdown below.",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llrrrrrrrr}",
        r"  \toprule",
        r"  Domain & Split & \multicolumn{2}{c}{CGCNN old} & \multicolumn{2}{c}{CGCNN new} & "
        r"\multicolumn{2}{c}{Trans.\ old} & \multicolumn{2}{c}{Trans.\ new} \\",
        r"  & & mean & med. & mean & med. & mean & med. & mean & med. \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for split in ("all", "validation"):
            o = payload["blocks"]["old"][domain][split]
            n = payload["blocks"]["new"][domain][split]
            split_label = "all" if split == "all" else "validation"
            lines.append(
                f"  {domain.capitalize():6s} & {split_label:10s} & "
                f"{fmt(o['abs_err_cgcnn_mean'], 2)} & {fmt(o['abs_err_cgcnn_median'])} & "
                f"{fmt(n['abs_err_cgcnn_mean'], 2)} & {fmt(n['abs_err_cgcnn_median'])} & "
                f"{fmt(o['abs_err_transformer_mean'], 2)} & {fmt(o['abs_err_transformer_median'])} & "
                f"{fmt(n['abs_err_transformer_mean'], 2)} & {fmt(n['abs_err_transformer_median'])} \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\noindent\textit{Note:} For point validation CGCNN, old/new medians are "
        f"{fmt(payload['blocks']['old']['point']['validation']['abs_err_cgcnn_median'])} / "
        f"{fmt(payload['blocks']['new']['point']['validation']['abs_err_cgcnn_median'])}~eV "
        r"while per-atom MAE remains $\sim 0.003$~eV.",
        r"",
        r"\section*{Table 3 | Relative total error $R_{\mathrm{tot}}$ (\%)}",
        r"",
        r"Crystals with $|\Delta E_{\mathrm{true}}|\ge 10^{-6}$~eV only ($n_{\mathrm{point}}=422$, "
        r"$n_{\mathrm{planar}}=1978$; 211 point configs with zero net change excluded).",
        r"",
        r"\begin{center}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll r rr rr}",
        r"  \toprule",
        r"  Domain & Split & $n$ & \multicolumn{2}{c}{CGCNN old/new med.} & \multicolumn{2}{c}{Trans.\ old/new med.} \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for split in ("all", "validation"):
            o = payload["blocks"]["old"][domain][split]
            n = payload["blocks"]["new"][domain][split]
            split_label = "all" if split == "all" else "validation"
            lines.append(
                f"  {domain.capitalize():6s} & {split_label:10s} & {o['n_rtot']} & "
                f"{fmt(o['rtot_cgcnn']['median'], 2)} & {fmt(n['rtot_cgcnn']['median'], 2)} & "
                f"{fmt(o['rtot_transformer']['median'], 2)} & {fmt(n['rtot_transformer']['median'], 2)} \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\subsection*{3a. Validation $R_{\mathrm{tot}}$ --- full statistics}",
        r"",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{ll rr rr rr rr}",
        r"  \toprule",
        r"  Domain & Model & \multicolumn{2}{c}{Old mean/med.} & \multicolumn{2}{c}{New mean/med.} & "
        r"\multicolumn{2}{c}{$<100\%$ old/new} \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        o = payload["blocks"]["old"][domain]["validation"]
        n = payload["blocks"]["new"][domain]["validation"]
        lines.append(
            f"  {domain.capitalize():6s} & CGCNN & "
            f"{fmt(o['rtot_cgcnn']['mean'], 1)} & {fmt(o['rtot_cgcnn']['median'], 1)} & "
            f"{fmt(n['rtot_cgcnn']['mean'], 1)} & {fmt(n['rtot_cgcnn']['median'], 1)} & "
            f"{fmt_pct(o['frac_rtot_c_below_100'])} & {fmt_pct(n['frac_rtot_c_below_100'])} \\\\"
        )
        lines.append(
            f"  & Transformer & "
            f"{fmt(o['rtot_transformer']['mean'], 1)} & {fmt(o['rtot_transformer']['median'], 1)} & "
            f"{fmt(n['rtot_transformer']['mean'], 1)} & {fmt(n['rtot_transformer']['median'], 1)} & "
            f"{fmt_pct(o['frac_rtot_t_below_100'])} & {fmt_pct(n['frac_rtot_t_below_100'])} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 4 | Paired crystal comparison (validation, $\Delta R_{\mathrm{tot}}$)}",
        r"",
        r"\noindent Of 63 point validation crystals, 21 have $|\Delta E_{\mathrm{true}}|<10^{-6}$~eV and are "
        r"excluded from $R_{\mathrm{tot}}$; paired $R_{\mathrm{tot}}$ comparisons use the remaining 42.",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"  \toprule",
        r"  Domain & Model & $n$ & improved & worsened & median $\Delta R_{\mathrm{tot}}$ (new$-$old) \\",
        r"  \midrule",
    ]
    for domain in ("point", "planar"):
        for metric, label in [
            ("rtot_cgcnn", "CGCNN"),
            ("rtot_transformer", "Transformer"),
        ]:
            p = payload["paired"][domain]["validation"][metric]
            lines.append(
                f"  {domain.capitalize():6s} & {label} & {p['n']} & "
                f"{p['improved']} ({p['pct_improved']:.0f}\\%) & {p['worsened']} & "
                f"{p['median_delta']:+.1f} pp \\\\"
            )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 5 | Point validation: absolute net error by $|\Delta E_{\mathrm{true}}|$ bucket}",
        r"",
        r"Explains why a median $|\mathrm{error}| \sim 0.11$~eV coexists with MAE $\sim 0.003$~eV.",
        r"",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{l r rr rr}",
        r"  \toprule",
        r"  Bucket & $n$ & \multicolumn{2}{c}{Old CGCNN med.\ $|\mathrm{err}|$ / MAE} & "
        r"\multicolumn{2}{c}{New CGCNN med.\ $|\mathrm{err}|$ / MAE} \\",
        r"  \midrule",
    ]
    for b_old, b_new in zip(
        payload["buckets"]["old"]["point"]["validation"],
        payload["buckets"]["new"]["point"]["validation"],
    ):
        lines.append(
            f"  {b_old['bucket']} & {b_old['n']} & "
            f"{fmt(b_old['abs_err_cgcnn_median'])} / {fmt(b_old['mae_cgcnn_mean'])} & "
            f"{fmt(b_new['abs_err_cgcnn_median'])} / {fmt(b_new['mae_cgcnn_mean'])} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Table 6 | Largest validation outliers (new CGCNN, point)}",
        r"",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{l r r r r}",
        r"  \toprule",
        r"  Config & $\Delta E_{\mathrm{true}}$ (eV) & MAE (eV) & $|\mathrm{err}|$ (eV) & $R_{\mathrm{tot}}$ (\%) \\",
        r"  \midrule",
    ]
    for row in payload["outliers"]["new"]["point"]["validation"]:
        label = latex_escape_texttt(f"{row['folder']}/{row['config']}")
        lines.append(
            f"  \\texttt{{{label}}} & {fmt(row['delta_true_eV'])} & "
            f"{fmt(row['mae_cgcnn_eV'])} & {fmt(row['abs_err_cgcnn_eV'])} & "
            f"{fmt(row['rtot_cgcnn_pct'], 1)} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"",
        r"\section*{Bottom line}",
        r"\begin{itemize}",
        r"  \item \textbf{Per-atom MAE} changed little on point validation "
        f"({fmt(payload['blocks']['old']['point']['validation']['mae_cgcnn_mean'])} $\\rightarrow$ "
        f"{fmt(payload['blocks']['new']['point']['validation']['mae_cgcnn_mean'])}~eV CGCNN).",
        r"  \item \textbf{Net energy ($R_{\mathrm{tot}}$)} on point validation: CGCNN median improved "
        f"({fmt(payload['blocks']['old']['point']['validation']['rtot_cgcnn']['median'], 1)}\\% "
        f"$\\rightarrow$ {fmt(payload['blocks']['new']['point']['validation']['rtot_cgcnn']['median'], 1)}\\%); "
        f"transformer median {fmt(payload['blocks']['old']['point']['validation']['rtot_transformer']['median'], 1)}\\% "
        f"$\\rightarrow$ {fmt(payload['blocks']['new']['point']['validation']['rtot_transformer']['median'], 1)}\\% "
        r"(similar). Paired on 42 crystals with nonzero net change: CGCNN improved on "
        f"{payload['paired']['point']['validation']['rtot_cgcnn']['pct_improved']:.0f}\\% of cases.",
        r"  \item \textbf{Planar validation} CGCNN median $R_{\mathrm{tot}}$ improved "
        f"({fmt(payload['blocks']['old']['planar']['validation']['rtot_cgcnn']['median'], 1)}\\% "
        f"$\\rightarrow$ {fmt(payload['blocks']['new']['planar']['validation']['rtot_cgcnn']['median'], 1)}\\%); "
        r"transformer median similar, mean much better (fewer catastrophic outliers).",
        r"  \item The headline 0.11~eV is a \textbf{median absolute net error}, not per-atom error. "
        r"It is dominated by crystals with modest true net changes where a $\sim$0.1~eV bias is "
        r"moderate in $R_{\mathrm{tot}}$ but looks large in eV.",
        r"  \item Means over all point crystals remain inflated by rare large-$N$ cells with large "
        r"$|\mathrm{err}|$ (see outlier table); prefer validation medians and $R_{\mathrm{tot}}$ for reporting.",
        r"\end{itemize}",
        r"",
        r"\end{document}",
        r"",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions vs predictions_new.")
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--latex-out", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()

    print("Loading old delivery …", flush=True)
    old_rows = load_delivery(args.old_root)
    print("Loading new delivery …", flush=True)
    new_rows = load_delivery(args.new_root)

    blocks: Dict[str, Dict[str, Dict[str, dict]]] = {"old": {}, "new": {}}
    for label, rows in (("old", old_rows), ("new", new_rows)):
        blocks[label] = {}
        for domain in ("point", "planar"):
            blocks[label][domain] = {}
            for split in ("all", "validation"):
                subset = [r for r in rows if r.domain == domain]
                if split == "validation":
                    subset = [r for r in subset if r.split == "validation"]
                stats = block_stats(label, subset, domain=domain, split=split)
                blocks[label][domain][split] = asdict(stats)

    paired: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for domain in ("point", "planar"):
        paired[domain] = {"validation": {}}
        for metric in (
            "rtot_cgcnn",
            "rtot_transformer",
            "mae_cgcnn",
            "mae_transformer",
            "abs_err_cgcnn",
            "abs_err_transformer",
        ):
            paired[domain]["validation"][metric] = paired_improvements(
                old_rows, new_rows, domain=domain, split="validation", metric=metric
            )

    buckets = {"old": {}, "new": {}}
    outliers = {"old": {}, "new": {}}
    for label, rows in (("old", old_rows), ("new", new_rows)):
        buckets[label] = {}
        outliers[label] = {}
        for domain in ("point", "planar"):
            buckets[label][domain] = {
                "validation": delta_bucket_stats(rows, domain=domain, split="validation")
            }
            outliers[label][domain] = {
                "validation": top_outliers(rows, domain=domain, split="validation")
            }

    train_path = ROOT / "delivery_global_v2_curves.json"
    train = {}
    if train_path.is_file():
        curves = json.loads(train_path.read_text(encoding="utf-8"))
        for k, r in curves.get("runs", {}).items():
            train[k] = {
                "lambda_tot": r["lambda_tot"],
                "best_val_r_tot_median": r["best_val_r_tot_median"],
                "best_val_mae": r["best_val_mae"],
            }

    payload = {
        "old_root": str(args.old_root),
        "new_root": str(args.new_root),
        "min_delta_eV": MIN_DELTA,
        "training_val_checkpoint": train,
        "blocks": blocks,
        "paired": paired,
        "buckets": buckets,
        "outliers": outliers,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_latex(payload, args.latex_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.latex_out}")

    o = blocks["old"]["point"]["validation"]
    n = blocks["new"]["point"]["validation"]
    print("\nPoint validation quick summary:")
    print(
        f"  MAE CGCNN:        {o['mae_cgcnn_mean']:.4f} -> {n['mae_cgcnn_mean']:.4f} eV"
    )
    print(
        f"  |err| CGCNN med:  {o['abs_err_cgcnn_median']:.4f} -> {n['abs_err_cgcnn_median']:.4f} eV"
    )
    print(
        f"  R_tot CGCNN med:  {o['rtot_cgcnn']['median']:.1f}% -> {n['rtot_cgcnn']['median']:.1f}%"
    )


if __name__ == "__main__":
    main()
