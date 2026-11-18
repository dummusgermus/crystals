"""GRACE-only training (5-fold CV by compound): pool the test predictions of the 5 folds
(every compound is tested exactly once) and compare with the zero-shot models.
Writes grace_train/analysis.{md,json}."""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.environ.get("GRACE_TRAIN_DIR", os.path.dirname(os.path.abspath(__file__)))
R = torch.load(os.path.join(ROOT, "grace_eval", "graphs_relax.pt"), weights_only=False)
ZS = torch.load(os.path.join(ROOT, "grace_eval", "predictions.pt"), weights_only=False)
TYPES = ("FFHFF", "Primitive", "XYZ")


def metrics(pred_tot, idx):
    """pred_tot: dict graph index -> predicted total dE (eV)."""
    out = {}
    for t in TYPES:
        ii = [i for i in idx if R[i].stack_sequence == t and i in pred_tot]
        n = np.array([R[i].num_nodes for i in ii]); tt = np.array([float(R[i].y.sum()) for i in ii])
        pt = np.array([pred_tot[i] for i in ii]); err = np.abs(pt - tt) / n * 1000
        ok = np.abs(tt) / n * 1000 > 0.5
        out[t] = {"n": len(ii), "tot_err_median": float(np.median(err)), "tot_err_mean": float(np.mean(err)),
                  "r_tot_median": float(np.median(100 * np.abs(pt - tt)[ok] / np.abs(tt)[ok])) if ok.any() else None,
                  "beats_zero": float(np.mean(np.abs(pt - tt) < np.abs(tt))) if t != "XYZ" else None,
                  "zero_baseline_median": float(np.median(np.abs(tt) / n * 1000))}
    return out


def main():
    runs = defaultdict(dict)  # (model, n) -> {(fold, seed): {graph idx: (pred_tot_rtot, pred_tot_mae, atom preds)}}
    for p in glob.glob(os.path.join(HERE, "preds", "*.pt")):
        m = re.search(r"cv5_planar_(cgcnn|transformer)_.*_s(\d+)_f(\d)-n(\w+)\.pt", p)
        d = torch.load(p, weights_only=False)
        for ck in ("ckpt_rtot", "ckpt_mae"):
            t = d[ck]["test"]; g = t["graph"].numpy(); pr = t["pred"].numpy(); idx = t["indices"].numpy()
            tot = np.bincount(g, pr, len(idx))
            runs[(m[1], m[4], ck)][(int(m[3]), int(m[2]))] = dict(zip(idx.tolist(), tot.tolist()))
    res = {}
    for (model, n, ck), fs in sorted(runs.items()):
        seeds = sorted({s for _, s in fs}); folds = sorted({f for f, _ in fs})
        per_seed = []
        for s in seeds:
            pooled = {}
            for f in folds:
                pooled.update(fs.get((f, s), {}))
            per_seed.append(metrics(pooled, list(pooled)))
        ens = defaultdict(list)
        for (f, s), d in fs.items():
            for i, v in d.items():
                ens[i].append(v)
        ens_m = metrics({i: float(np.mean(v)) for i, v in ens.items()}, list(ens))
        avg = {t: {k: (float(np.mean([ps[t][k] for ps in per_seed])) if per_seed[0][t][k] is not None else None)
                   for k in per_seed[0][t]} for t in TYPES}
        res[f"{model}/n{n}/{ck}"] = {"n_runs": len(fs), "folds": len(folds), "seeds": len(seeds), "single_avg": avg, "seed_ensemble": ens_m}
    all_idx = list(range(len(R)))
    for m in ("cgcnn-fast", "cgcnn-ensemble5", "transformer-fast", "transformer-ensemble5"):
        res[f"zero-shot/{m}"] = metrics({i: float(ZS[m]["relax"][i].sum()) for i in all_idx}, all_idx)
    json.dump(res, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)
    L = ["# GRACE-only training vs. zero-shot (all 189 compounds, each tested once via 5-fold CV)\n",
         "Median error of the total relaxation energy (meV/atom); R_tot over structures with |dE| > 0.5 meV/atom; "
         "'beats 0' = share of compounds better than predicting no relaxation. Trained: checkpoint = best val R_tot, mean over 3 seeds "
         "(seed ens. = mean prediction of the 3 seeds).\n"]
    for t in TYPES:
        L += [f"\n## {t}\n", "| model | train compounds | median err | R_tot % | beats 0 | mean err |", "|---|---|---|---|---|---|"]
        zb = res["zero-shot/transformer-fast"][t]["zero_baseline_median"]
        L.append(f"| no relaxation | – | {zb:.2f} | 100 | – | |")
        for m in ("cgcnn-ensemble5", "transformer-ensemble5"):
            v = res[f"zero-shot/{m}"][t]
            L.append(f"| zero-shot {m} | 0 (old data) | {v['tot_err_median']:.2f} | {v['r_tot_median'] or float('nan'):.1f} | {v['beats_zero'] if v['beats_zero'] is None else round(v['beats_zero'], 2)} | {v['tot_err_mean']:.1f} |")
        for model in ("cgcnn", "transformer"):
            for n in ("25", "50", "100", "all"):
                for kind in ("single_avg", "seed_ensemble"):
                    k = f"{model}/n{n}/ckpt_rtot"
                    if k not in res:
                        continue
                    v = res[k][kind][t]
                    L.append(f"| GRACE-trained {model}{' (seed ens.)' if kind == 'seed_ensemble' else ''} | {n} | {v['tot_err_median']:.2f} | "
                             f"{(v['r_tot_median'] if v['r_tot_median'] is not None else float('nan')):.1f} | "
                             f"{'' if v['beats_zero'] is None else round(v['beats_zero'], 2)} | {v['tot_err_mean']:.1f} |")
    open(os.path.join(HERE, "analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
