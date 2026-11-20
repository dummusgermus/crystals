"""Aggregate displacement runs: mean over seeds, rank by validation r_disp.

    python displacement/analyze.py [runs_dir] [out_name]
Writes <runs_dir>/analysis_<out_name>.md and .json.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KEYS = ("r_disp", "skill", "mae", "cos", "r_disp_all", "skill_all", "cell_mae", "null_max_pred")


def force_baseline(domain, view):
    """d = c * F0 with one c fit on the training split (same split / region as the runs)."""
    from train_disp import load_views, metrics, parse_spec, split_by_compound
    spec = parse_spec(view)
    ds = load_views(domain, spec)
    tr, va, te = split_by_compound(ds, 0)
    F = torch.cat([ds[i].force0[ds[i].loss_mask] for i in tr if not bool(ds[i].unstable)])
    D = torch.cat([ds[i].y[ds[i].loss_mask] for i in tr if not bool(ds[i].unstable)])
    c = float((F * D).sum() / (F * F).sum())
    out = {}
    for name, idx in (("val", va), ("test", te)):
        items = [dict(pred=c * ds[i].force0 - (c * ds[i].force0.mean(0) if domain == "planar" else 0),
                      y=ds[i].y, mask=ds[i].loss_mask, unstable=bool(ds[i].unstable), null=bool(ds[i].null),
                      folder=ds[i].folder, key=ds[i].key) for i in idx]
        out[name] = metrics(items)
    out["c"] = c
    return out


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
    name = sys.argv[2] if len(sys.argv) > 2 else "phase1"
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(runs, "results", "*.json")):
        r = json.load(open(p))
        cfg = r["config"]
        k = "_".join(r["run_key"].split("_")[:-2])  # drop _s{seed}_sp{split}
        groups[(cfg["domain"], cfg["model"], k)].append(r)
    rows = []
    for (dom, model, k), rs in groups.items():
        row = dict(domain=dom, model=model, key=k, n=len(rs), seeds=sorted(r["config"]["seed"] for r in rs),
                   best_epoch=float(np.mean([r["best_epoch"] for r in rs])),
                   time_min=float(np.mean([r["train_time_s"] for r in rs]) / 60),
                   view=rs[0]["config"]["spec"], head=rs[0]["config"]["head"], loss=rs[0]["config"]["loss"],
                   head_force=rs[0]["config"]["head_force"])
        for part in ("val", "test"):
            for m in KEYS:
                vals = [r[part].get(m) for r in rs if r[part].get(m) is not None]
                row[f"{part}_{m}"] = float(np.mean(vals)) if vals else None
                row[f"{part}_{m}_sd"] = float(np.std(vals)) if len(vals) > 1 else None
        rows.append(row)
    base = {}
    for dom, view in (("point", "rt=8.9,buf=4,edge=k3"), ("planar", "edge=k3")):
        try:
            base[dom] = force_baseline(dom, view)
        except Exception as e:  # noqa: BLE001
            base[dom] = {"error": repr(e)}
    json.dump(dict(rows=rows, force_baseline=base), open(os.path.join(runs, f"analysis_{name}.json"), "w"), indent=1)

    f = lambda v, p=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"  # noqa: E731
    L = [f"# Displacement study: {name}\n",
         "r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); "
         "skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; "
         "ranked by validation r_disp. Test split = unseen compounds.\n"]
    for dom in ("point", "planar"):
        b = base.get(dom, {})
        if "test" in b:
            L.append(f"\n**{dom} reference 'd = c * F0'** (c = {b['c']:.4f} A^2/eV): val r_disp {f(b['val']['r_disp'])}, "
                     f"test r_disp {f(b['test']['r_disp'])}, test skill {f(b['test']['skill'])}, test mae {f(b['test']['mae'], 4)} "
                     f"(zero-motion mae {f(b['test']['zero_mae'], 4)})\n")
        for model in ("cgcnn", "transformer", "painn"):
            rs = sorted([r for r in rows if r["domain"] == dom and r["model"] == model],
                        key=lambda r: r["val_r_disp"] if r["val_r_disp"] is not None else 9)
            if not rs:
                continue
            L += [f"\n## {dom} · {model}\n",
                  "| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for i, r in enumerate(rs, 1):
                v = r["view"]
                vs = (f"rt{v['rt']:g}/buf{v['buf']:g} " if dom == "point" else "") + f"{v['edge']} {v['feat']}" + \
                     (" stable-only" if v["stable"] else "")
                L.append(f"| {i} | {vs} | {r['head']}{'+F' if r['head_force'] else ''} | {r['loss']} | {r['n']} | "
                         f"{f(r['val_r_disp'])} ± {f(r['val_r_disp_sd'], 2)} | {f(r['test_r_disp'])} | {f(r['test_skill'])} | "
                         f"{f(r['test_mae'], 4)} | {f(r['test_cos'], 2)} | {f(r['test_r_disp_all'])} | {f(r['test_cell_mae'], 4)} | "
                         f"{f(r['test_null_max_pred'], 4)} | {r['best_epoch']:.0f} | {r['time_min']:.0f} |")
    open(os.path.join(runs, f"analysis_{name}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
