"""Energy study summary (runs_energy vs position-only joint PaiNN in runs_joint).
    python displacement/energy_study.py      -> prints table, writes runs_energy/energy_study.json
Per config: position metrics = mean over the 6 runs (3 splits x 2 seeds) of the test JSONs; energy metrics
from the saved test predictions, seed-ensembled per split, then averaged over splits:
  head      learned energy head (sum of per-atom predictions)
  harm      harmonic estimate -1/2 sum F0 . d_pred from the same model's displacements
  blend     0.5 (head + harm)
"""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from harmonic_energy import lookup  # noqa: E402

PRE = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_"
CFGS = [("position only", "runs_joint", "dfrecipe"), ("energy only", "runs_energy", "ew1_pw0_dfrecipe"),
        ("multi-task ew0.3", "runs_energy", "ew0.3_dfrecipe"), ("multi-task ew1", "runs_energy", "ew1_dfrecipe"),
        ("multi-task ew3", "runs_energy", "ew3_dfrecipe"), ("multi-task ew1 + F0.d", "runs_energy", "ew1_ec_dfrecipe")]


def emet(p, t):
    err = np.abs(p - t); big = np.abs(t) > 1e-3
    return dict(e_rel=float(np.median(err[big] / np.abs(t[big]))), e_mae=float(err.mean()),
                e_skill=float(1 - (err ** 2).sum() / (t ** 2).sum()))


def main():
    L = None; res = {}
    for name, rd, tag in CFGS:
        js = [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/{rd}/results/{PRE}{tag}_s*_sp*.json"))]
        r = {f"{d}_{k}": float(np.mean([j["test"][f"{d}_{k}"] for j in js])) for d in ("point", "planar") for k in ("r_disp", "skill")}
        r["n_runs"] = len(js)
        acc = {}
        for sp in (0, 1, 2):
            P = [torch.load(p, weights_only=False, map_location="cpu") for p in sorted(glob.glob(f"{HERE}/{rd}/preds/{PRE}{tag}_s*_sp{sp}.pt"))]
            if L is None:
                L = lookup(P[0]["config"]["spec"])
            rows = {0: [], 1: []}
            for its in zip(*[sorted(x["test"], key=lambda i: (i["folder"], str(i["key"]))) for x in P]):
                it = its[0]
                if it["unstable"]:
                    continue
                dom = int(it["dom"]); F0, de = L[("point" if dom == 0 else "planar", it["folder"], str(it["key"]))]
                d = torch.stack([i["pred"] for i in its]).mean(0).double()
                harm = float(-0.5 * (F0 * d).sum())
                head = float(np.mean([float(i["e_pred"].sum()) for i in its])) if "e_pred" in it else np.nan
                rows[dom].append((head, harm, float(de.sum())))
            for dom, dn in ((0, "point"), (1, "planar")):
                a = np.array(rows[dom])
                for est, col in (("head", a[:, 0]), ("harm", a[:, 1]), ("blend", 0.5 * (a[:, 0] + a[:, 1]))):
                    if np.isnan(col).any():
                        continue
                    for k, v in emet(col, a[:, 2]).items():
                        acc.setdefault(f"{dn}_{est}_{k}", []).append(v)
        r.update({k: float(np.mean(v)) for k, v in acc.items()})
        res[name] = r
        print(f"{name:24s} r_disp pt {r['point_r_disp']:.3f} pl {r['planar_r_disp']:.3f} | " + " | ".join(
            f"{est}: pt {r.get(f'point_{est}_e_rel', np.nan):.3f}/{r.get(f'point_{est}_e_mae', np.nan):.3f}eV "
            f"pl {r.get(f'planar_{est}_e_rel', np.nan):.3f}/{1e3*r.get(f'planar_{est}_e_mae', np.nan):.1f}meV"
            for est in ("head", "harm", "blend")), flush=True)
    json.dump(res, open(f"{HERE}/runs_energy/energy_study.json", "w"), indent=1)


if __name__ == "__main__":
    main()
