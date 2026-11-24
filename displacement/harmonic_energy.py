"""Relaxation energy from predicted displacements: dE ~ -1/2 sum_i F0_i . d_i (harmonic estimate).
    python displacement/harmonic_energy.py <preds prefix> [...]      (prefix = preds/<config> without _s*_sp*.pt)
Scores test sets (stable structures) per domain; several prefixes -> ensemble mean displacement per split.
"""
import glob, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from views import make_view, parse_spec  # noqa: E402

def lookup(spec):
    L = {}
    for d in ("point", "planar"):
        for g in torch.load(os.path.join(HERE, "data", f"superset_{d}.pt"), weights_only=False):
            v = make_view(g, spec)
            L[(d, g.folder, str(v.key))] = (v.force0.double(), v.de.double())
    return L

def score(prefixes, part="test"):
    spec = None; out = {0: [], 1: []}
    for sp in (0, 1, 2):
        P = [torch.load(p, weights_only=False) for pre in prefixes for p in sorted(glob.glob(f"{pre}_s*_sp{sp}.pt"))]
        if spec is None:
            spec = P[0]["config"]["spec"]; L = lookup(spec)
        items = [sorted(x[part], key=lambda i: (i["folder"], str(i["key"]))) for x in P]
        for its in zip(*items):
            it = its[0]
            if it["unstable"]:
                continue
            dom = it.get("dom", 1 if it["folder"].count("_") and not it["folder"].startswith("C1") else 0)
            F0, de = L[("point" if dom == 0 else "planar", it["folder"], str(it["key"]))]
            d = torch.stack([i["pred"] for i in its]).mean(0).double()
            out[dom].append((float(-0.5 * (F0 * d).sum()), float(de.sum())))
    res = {}
    for k, name in ((0, "point"), (1, "planar")):
        a = np.array(out[k]); err = np.abs(a[:, 0] - a[:, 1]); big = np.abs(a[:, 1]) > 1e-3
        res[name] = (float(np.median(err[big] / np.abs(a[big, 1]))), float(err.mean()), len(a))
    return res

if __name__ == "__main__":
    r = score(sys.argv[1:])
    for k, (rel, mae, n) in r.items():
        print(f"{k}: e_rel {rel:.3f}  MAE {mae:.4f} eV  (n={n})")
