"""Per-crystal inference latency (batch size 1, like crystal_prediction_export): single vs. N-seed ensemble.
    python production_retrain/bench_ensemble.py [n_seeds] [device]"""
import json, os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from crystal_prediction_export import GLOBAL_V2_JOBS, load_residual_model, predict_residual_delta
from delivery_global_v2 import CHECKPOINTS, TOTALS_DATASETS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu"))
out = {}
for key, job in GLOBAL_V2_JOBS.items():
    domain, kind = key.split("_", 1)
    ds = torch.load(TOTALS_DATASETS[domain], weights_only=False)
    val = [ds[i] for i in torch.load(CHECKPOINTS[key], map_location="cpu", weights_only=False)["val_indices"]][:60]
    paths = [os.path.join(ROOT, "production_retrain", "ckpts", f"{key}_s{s}.pt") for s in range(N)]
    singles = [load_residual_model(p, val, kind, dev) for p in paths]
    ens = load_residual_model(",".join(paths), val, kind, dev)
    def run(m):
        ts, preds = [], []
        for rep in range(2):  # first pass = warm-up
            ts, preds = [], []
            for g in val:
                p, _, t = predict_residual_delta(m[0], g, dev, m[1], m[2], False)
                ts.append(t); preds.append(p)
        return np.array(ts), preds
    t1, p1 = run(singles[0])
    allp = [run(s)[1] for s in singles]
    tE, pE = run(ens)
    err = max(float(np.abs(pE[i] - np.mean([a[i] for a in allp], 0)).max()) for i in range(len(val)))
    out[key] = {"single_ms": 1000 * float(np.median(t1)), "ensemble_ms": 1000 * float(np.median(tE)),
                "ratio": float(np.median(tE) / np.median(t1)), "max_abs_diff_vs_mean_eV": err,
                "atoms_median": int(np.median([g.num_nodes for g in val]))}
    print(key, out[key], flush=True)
json.dump(out, open(os.path.join(ROOT, "production_retrain", f"bench_ensemble_{N}_{dev.type}.json"), "w"), indent=1)
