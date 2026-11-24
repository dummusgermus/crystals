"""Aggregate the benchmark runs (runs_bench/results) -> table (mean +- std over 3 seeds) + bench_summary.json."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "runs_bench", "results")
def arm(c):
    if c["model"] != "painn":
        return f"{c['model']}" + ("+ours" if c.get("ext_feats") else " (published)")
    if c.get("bench_feats") == "none":
        return "painn no-GRACE"
    return "painn+ours dfrecipe" if c.get("tag") == "dfrecipe" else "painn+ours"
rows = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f)); c = d["config"]
    ds = c["domain"].split(":")[1]
    rows.setdefault((ds, arm(c)), []).append(d)
grace = json.load(open(os.path.join(os.path.dirname(HERE), "bench_data", "grace_baseline.json")))
out = {}
for (ds, a), rs in sorted(rows.items()):
    t = {k: [r["test"][k] for r in rs] for k in ("mae_pos", "mae_metric", "mae_vol", "r_disp")}
    v = [r["val"]["mae_pos"] for r in rs]
    out.setdefault(ds, {"dummy": {k: rs[0]["test"][k + "_dummy"] for k in ("mae_pos", "mae_metric", "mae_vol")},
                        "grace": grace.get(ds), "arms": {}})
    out[ds]["arms"][a] = {**{k: [float(np.mean(x)), float(np.std(x))] for k, x in t.items()},
                          "val_mae_pos": float(np.mean(v)), "n": len(rs), "skipped": sum(r["n_skipped"] for r in rs),
                          "best_ep": [r["best_epoch"] for r in rs]}
for ds, o in out.items():
    dm, g = o["dummy"], o["grace"] or {}
    print(f"\n{ds}: dummy pos {dm['mae_pos']:.4f} metric {dm['mae_metric']:.2f} vol {dm['mae_vol']:.2f} | "
          f"GRACE-relax pos {g.get('mae_pos', float('nan')):.4f} metric {g.get('mae_metric', float('nan')):.2f} "
          f"vol {g.get('mae_vol', float('nan')):.2f} r_disp {g.get('r_disp', float('nan')):.3f}")
    for a, s in sorted(o["arms"].items(), key=lambda kv: kv[1]["mae_pos"][0]):
        print(f"  {a:22s} pos {s['mae_pos'][0]:.4f}+-{s['mae_pos'][1]:.4f} metric {s['mae_metric'][0]:6.2f} "
              f"vol {s['mae_vol'][0]:7.2f} r_disp {s['r_disp'][0]:.3f} | val pos {s['val_mae_pos']:.4f} "
              f"n={s['n']} skip={s['skipped']} ep={s['best_ep']}")
json.dump(out, open(os.path.join(os.path.dirname(HERE), "bench_data", "bench_summary.json"), "w"), indent=1)
