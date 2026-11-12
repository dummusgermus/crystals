"""Aggregate atom_encoding_sweep/results/*.json -> summary.json + console tables.

Metrics (test set, checkpoint = best val R_tot, i.e. production checkpointing):
  mae     : per-atom MAE (eV/atom)
  rtot    : median relative net-energy error (%), production target mode
  dE      : mean |sum pred - sum true| (eV)
plus mae_bestmae: test MAE at the best-val-MAE checkpoint.
Paired deltas are vs. type12 with identical split + seed.
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENC_ORDER = [
    "type12", "none", "role_onehot", "elem_onehot",
    "elem_props", "elem_props_role", "pair_props", "pair_props_edge", "onehot_props",
]


def load():
    runs = defaultdict(dict)  # (split, domain, model, enc) -> seed -> metrics
    for path in glob.glob(os.path.join(HERE, "results", "*.json")):
        r = json.load(open(path))
        t = r["ckpt_rtot"]["test"]
        runs[(r["split"], r["domain"], r["model_kind"], r["encoding"])][r["seed"]] = {
            "mae": t["mae"],
            "rtot": t["r_tot_median"],
            "dE": t["abs_total_err_eV"],
            "mae_bestmae": r["ckpt_mae"]["test"]["mae"],
            "val_rtot": r["ckpt_rtot"]["val"]["r_tot_median"],
        }
    return runs


def main():
    runs = load()
    summary = {}
    for split in ("random", "chem"):
        for domain in ("point", "planar"):
            for model in ("cgcnn", "transformer"):
                base = runs.get((split, domain, model, "type12"), {})
                rows = []
                for enc in ENC_ORDER:
                    seeds = runs.get((split, domain, model, enc), {})
                    if not seeds:
                        continue
                    agg = {"n_seeds": len(seeds)}
                    for m in ("mae", "rtot", "dE", "mae_bestmae", "val_rtot"):
                        v = np.array([s[m] for s in seeds.values()])
                        agg[m] = float(v.mean())
                        agg[m + "_std"] = float(v.std())
                        common = sorted(set(seeds) & set(base))
                        if enc != "type12" and common:
                            d = np.array([seeds[k][m] - base[k][m] for k in common])
                            agg["d_" + m] = float(d.mean())
                            agg["wins_" + m] = int((d < 0).sum())
                            agg["n_paired"] = len(common)
                    rows.append((enc, agg))
                    summary[f"{split}/{domain}/{model}/{enc}"] = agg
                if not rows:
                    continue
                print(f"\n### split={split} | {domain} | {model}")
                print(f"{'encoding':16s} n  {'test MAE':>15s} {'dMAE':>9s}  {'test R_tot%':>12s} {'dR_tot':>7s} wins  {'|dE| eV':>8s} {'d|dE|':>7s}  MAE@maeckpt")
                for enc, a in rows:
                    dm = f"{a.get('d_mae', 0):+.5f}" if "d_mae" in a else ""
                    dr = f"{a.get('d_rtot', 0):+6.1f}" if "d_rtot" in a else ""
                    w = f"{a.get('wins_rtot','')}/{a.get('n_paired','')}" if "d_rtot" in a else ""
                    dd = f"{a.get('d_dE', 0):+.3f}" if "d_dE" in a else ""
                    print(
                        f"{enc:16s} {a['n_seeds']}  {a['mae']:.5f}±{a['mae_std']:.5f} {dm:>9s}  "
                        f"{a['rtot']:5.1f}±{a['rtot_std']:4.1f} {dr:>7s} {w:>4s}  {a['dE']:8.3f} {dd:>7s}  {a['mae_bestmae']:.5f}"
                    )
    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
