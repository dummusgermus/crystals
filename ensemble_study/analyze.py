"""Ensemble study: combine per-atom test predictions of models trained on the SAME split
(same split seed) with different init seeds.

    python ensemble_study/analyze.py [--dir ensemble_study]

Compared (test set, checkpoint = best val R_tot unless stated):
  single      : mean over individual members (what one training run gives you)
  best-val    : the member with the lowest val R_tot (= "pick the best seed", like production)
  ens-k       : mean prediction of k members, averaged over all k-subsets (k = 2..K)
  ens-median  : median prediction of all K members
  ens-maeckpt : mean of all K members, best-val-MAE checkpoints
  ens-both    : mean of all 2K checkpoints (R_tot + MAE checkpoints)
  cross       : CGCNN ensemble and Transformer ensemble averaged 50/50 (same split)
Writes <dir>/analysis.{md,json}.
"""
import argparse, glob, itertools, json, os, re
from collections import defaultdict
import numpy as np
import torch

def metrics(pred, d, mode):
    g = d["graph"].numpy(); y = d["y"].numpy(); n = int(g.max()) + 1
    cnt = np.bincount(g, minlength=n)
    mae = float((np.bincount(g, np.abs(pred - y), n) / cnt).mean())
    s = np.bincount(g, pred, n); tot = d[f"tot_{mode}"].numpy()
    err = np.abs(s - tot); ok = np.abs(tot) >= 1e-6
    return {"mae": mae, "rtot": float(np.median(100 * err[ok] / np.abs(tot[ok]))), "dE": float(err.mean())}

def evaluate(pred, d):
    a = metrics(pred, d, "graph"); b = metrics(pred, d, "full")
    return {"mae": a["mae"], "rtot_graph": a["rtot"], "rtot_full": b["rtot"], "dE_graph": a["dE"], "dE_full": b["dE"]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    groups = defaultdict(dict)
    for p in glob.glob(os.path.join(a.dir, "preds", "*.pt")):
        m = re.match(r"(random|chem)_(point|planar)_(cgcnn|transformer)_.*_sp(\d+)_s(\d+)\.pt", os.path.basename(p))
        groups[(m[1], m[2], m[3], int(m[4]))][int(m[5])] = torch.load(p, weights_only=False)
    res, ens_full = {}, {}
    for key in sorted(groups):
        mem = groups[key]; seeds = sorted(mem); K = len(seeds)
        own = mem[seeds[0]]["own_target_mode"]
        T = {s: mem[s]["ckpt_rtot"]["test"] for s in seeds}; d = T[seeds[0]]
        for s in seeds:
            assert torch.equal(T[s]["indices"], d["indices"]) and torch.equal(T[s]["graph"], d["graph"])
        P = np.stack([T[s]["pred"].numpy() for s in seeds])
        Pm = np.stack([mem[s]["ckpt_mae"]["test"]["pred"].numpy() for s in seeds])
        single = [evaluate(P[i], d) for i in range(K)]
        val_rtot = [metrics(mem[s]["ckpt_rtot"]["val"]["pred"].numpy(), mem[s]["ckpt_rtot"]["val"], own)["rtot"] for s in seeds]
        r = {"K": K, "own_mode": own, "single": {m: float(np.mean([x[m] for x in single])) for m in single[0]},
             "single_all": single, "best-val": single[int(np.argmin(val_rtot))]}
        for k in range(2, K + 1):
            ev = [evaluate(P[list(c)].mean(0), d) for c in itertools.combinations(range(K), k)]
            r[f"ens-{k}"] = {m: float(np.mean([x[m] for x in ev])) for m in ev[0]}
        r["ens-median"] = evaluate(np.median(P, 0), d)
        r["ens-maeckpt"] = evaluate(Pm.mean(0), d)
        r["ens-both"] = evaluate(np.concatenate([P, Pm]).mean(0), d)
        res["/".join(map(str, key))] = r
        ens_full[key] = (P.mean(0), d)
    for (sp, dom, mdl, ss), (p, d) in ens_full.items():
        if mdl != "cgcnn" or (sp, dom, "transformer", ss) not in ens_full:
            continue
        p2, d2 = ens_full[(sp, dom, "transformer", ss)]
        assert torch.equal(d["indices"], d2["indices"])
        res[f"{sp}/{dom}/cross/{ss}"] = {"cross": evaluate(0.5 * (p + p2), d),
                                        "cgcnn-ens": res[f"{sp}/{dom}/cgcnn/{ss}"][f"ens-{res[f'{sp}/{dom}/cgcnn/{ss}']['K']}"],
                                        "transformer-ens": res[f"{sp}/{dom}/transformer/{ss}"][f"ens-{res[f'{sp}/{dom}/transformer/{ss}']['K']}"]}
    json.dump(res, open(os.path.join(a.dir, "analysis.json"), "w"), indent=1)
    write_md(res, a.dir)

def write_md(res, d):
    L = ["# Ensemble study (test set; % change vs. the average single model, same split)\n",
         "R_tot uses each model's own target mode (CGCNN point: full cell, others: graph sum). Negative = better.\n"]
    settings = sorted({tuple(k.split("/")[:3]) for k in res if k.split("/")[2] != "cross"}, key=lambda t: (t[2], t[1], t[0]))
    for sp, dom, mdl in settings:
        rows = [res[k] for k in res if k.startswith(f"{sp}/{dom}/{mdl}/")]
        K = rows[0]["K"]; rk = "rtot_full" if rows[0]["own_mode"] == "full" else "rtot_graph"
        L += [f"\n## {mdl} · {dom} · {sp} ({len(rows)} splits × {K} seeds)\n",
              "| method | MAE (meV) | ΔMAE % | R_tot % | ΔR_tot pp | |dE| (eV) | Δ|dE| % |", "|---|---|---|---|---|---|---|"]
        dE = "dE_full" if rk == "rtot_full" else "dE_graph"
        for meth in ["single", "best-val"] + [f"ens-{k}" for k in range(2, K + 1)] + ["ens-median", "ens-maeckpt", "ens-both"]:
            mae = np.mean([r[meth]["mae"] for r in rows]) * 1000; mae0 = np.mean([r["single"]["mae"] for r in rows]) * 1000
            rt = np.mean([r[meth][rk] for r in rows]); rt0 = np.mean([r["single"][rk] for r in rows])
            de = np.mean([r[meth][dE] for r in rows]); de0 = np.mean([r["single"][dE] for r in rows])
            L.append(f"| {meth} | {mae:.2f} | {100*(mae-mae0)/mae0:+.1f} | {rt:.2f} | {rt-rt0:+.2f} | {de:.3f} | {100*(de-de0)/de0:+.1f} |")
    cross = sorted({tuple(k.split("/")[:2]) for k in res if k.split("/")[2] == "cross"})
    if cross:
        L += ["\n## Cross-architecture: (CGCNN ensemble + Transformer ensemble) / 2\n",
              "| split | domain | method | MAE (meV) | R_tot graph % | R_tot full % |", "|---|---|---|---|---|---|"]
        for sp, dom in cross:
            rows = [res[k] for k in res if k.startswith(f"{sp}/{dom}/cross/")]
            for meth in ("cgcnn-ens", "transformer-ens", "cross"):
                L.append(f"| {sp} | {dom} | {meth} | {np.mean([r[meth]['mae'] for r in rows])*1000:.2f} | "
                         f"{np.mean([r[meth]['rtot_graph'] for r in rows]):.2f} | {np.mean([r[meth]['rtot_full'] for r in rows]):.2f} |")
    open(os.path.join(d, "analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
