"""Seed-ensemble metrics: average the saved predictions of runs that differ only in --seed.
    python displacement/ensemble.py <runs_dir> [<runs_dir> ...]
"""
import glob, os, re, sys
from collections import defaultdict
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_disp import metrics, R_EVAL_POINT  # noqa: E402

groups = defaultdict(list)
for d in sys.argv[1:]:
    for p in glob.glob(os.path.join(d, "preds", "*.pt")):
        k = os.path.basename(p)[:-3]
        base = re.sub(r"_s\d+_sp(\d+)$", r"_sp\1", k)
        groups[base].append(p)
rows = defaultdict(lambda: defaultdict(list))
for base, ps in sorted(groups.items()):
    if len(ps) < 2:
        continue
    P = [torch.load(p, weights_only=False) for p in ps]
    cfg = re.sub(r"_sp\d+$", "", base); planar = cfg.startswith("planar")
    r_eval = None if planar else R_EVAL_POINT
    for part in ("val", "test"):
        single = [metrics(x[part], r_eval)["r_disp"] for x in P]
        items = []
        for its in zip(*[x[part] for x in P]):
            it = dict(its[0]); it["pred"] = torch.stack([i["pred"] for i in its]).mean(0)
            if "cell_pred" in it and it["cell_pred"] is not None:
                it["cell_pred"] = torch.stack([i["cell_pred"] for i in its]).mean(0)
            items.append(it)
        m = metrics(items, r_eval)
        rows[cfg][part + "_single"].append(np.mean(single)); rows[cfg][part + "_ens"].append(m["r_disp"])
        rows[cfg][part + "_skill_ens"].append(m["skill"])
print(f"{'config':80s} {'val single':>10s} {'val ens':>8s} {'test single':>11s} {'test ens':>8s} {'test skill ens':>14s} splits")
for cfg, r in sorted(rows.items(), key=lambda kv: np.mean(kv[1]["val_ens"])):
    print(f"{cfg[:80]:80s} {np.mean(r['val_single']):10.3f} {np.mean(r['val_ens']):8.3f} {np.mean(r['test_single']):11.3f} "
          f"{np.mean(r['test_ens']):8.3f} {np.mean(r['test_skill_ens']):14.2f} {len(r['val_ens'])}")
