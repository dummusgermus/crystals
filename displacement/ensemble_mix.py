"""Cross-model ensemble: average saved predictions of several configs (all seeds) per split.
    python displacement/ensemble_mix.py <domain-tag> <preds_glob_prefix> [<preds_glob_prefix> ...]
Each prefix is a path to preds/<config> without _s{seed}_sp{split}.pt; e.g.
    displacement/runs_ext/preds/point_definet_..._ema  displacement/runs_p4/preds/point_painn_..._std
"""
import glob, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_disp import metrics, R_EVAL_POINT  # noqa: E402

planar = sys.argv[1] == "planar"; r_eval = None if planar else R_EVAL_POINT
prefixes = sys.argv[2:]
for part in ("val", "test"):
    res = []
    for sp in (0, 1, 2):
        P = [torch.load(p, weights_only=False)[part] for pre in prefixes for p in sorted(glob.glob(f"{pre}_s*_sp{sp}.pt"))]
        P = [[i for i in x if i.get("dom", int(planar)) == int(planar)] for x in P]  # joint models: this domain only
        keys = [sorted(range(len(x)), key=lambda i: (x[i]["folder"], str(x[i].get("key", "")))) for x in P]
        items = []
        for j in range(len(P[0])):
            its = [x[k[j]] for x, k in zip(P, keys)]
            assert all(i["folder"] == its[0]["folder"] and i["pred"].shape == its[0]["pred"].shape for i in its)
            it = dict(its[0]); it["pred"] = torch.stack([i["pred"] for i in its]).mean(0)
            cp = [i["cell_pred"] for i in its if i.get("cell_pred") is not None]  # DefiNet has no cell head
            if cp:
                it["cell_pred"] = torch.stack(cp).mean(0)
            items.append(it)
        m = metrics(items, r_eval); res.append((m["r_disp"], m["skill"], len(P)))
    print(part, "r_disp per split", [round(r[0], 3) for r in res], "mean %.3f" % np.mean([r[0] for r in res]),
          "skill %.2f" % np.mean([r[1] for r in res]), "members", [r[2] for r in res])
