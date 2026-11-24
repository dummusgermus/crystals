"""Score saved checkpoints (train_disp --save-preds files: state + config) on the mixed test cells.

    python displacement/mixed/eval_mixed.py <preds prefix> [...]   (prefix = preds/<config> without _s*_sp*.pt)
For every split seed sp, the members <prefix>_s*_sp{sp}.pt are averaged (seed ensemble) and scored on the mixed
cells whose compound is a test compound of union split sp. Non-unified (legacy joint) checkpoints see a mixed
cell as a planar cell whose defect channel is the union of fault atoms and the point defect (dist = nearest of
either): the best a single defect channel + domain flag can do.
Prints r_disp on all atoms / on the 8.9 A point-defect region / on the pure faults, the c-vector MAE and the
relaxation-energy error of -1/2 sum F0.d_pred (and of the energy head, when the members have one).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE)
sys.path.insert(0, DISP); sys.path.insert(0, os.path.dirname(DISP))
from predict_disp import build, views  # noqa: E402
from train_disp import R_EVAL_POINT, atom_energy, joint_gauge, metrics  # noqa: E402


def legacy(g):
    g = g.clone()
    g.domain = "planar"
    has_pd = bool((g.is_defect > 0.5).any())
    g.is_defect = torch.maximum(g.is_defect, g.is_fault)
    g.dist = torch.minimum(g.dist, g.dist_fault) if has_pd else g.dist_fault.clone()
    del g.is_fault, g.dist_fault
    return g


@torch.no_grad()
def run(ck, graphs, device):
    unified = bool(ck["config"].get("unified"))
    gs = graphs if unified else [legacy(g) for g in graphs]
    model, vs = build(ck, device), views(gs, ck)
    out = []
    for b in DataLoader(vs, batch_size=4):
        b = b.to(device)
        d, cell = model(b)
        dg = joint_gauge(model, b, d); dA = dg * b.dscale[b.batch].unsqueeze(-1)
        e = atom_energy(model, b, dg, ck["config"].get("e_harm", False), False) * b.edsc[b.batch] \
            if getattr(model, "e_atom", None) is not None else None
        for k in range(b.num_graphs):
            m = b.batch == k
            out.append(dict(pred=dA[m].cpu(), cell=(cell[k] * ck["cell_scale"]).cpu() if cell is not None else None,
                            e_head=float(e[m].sum()) if e is not None else None,
                            e_harm=float(-0.5 * (b.force0[m] * dA[m]).sum())))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = torch.load(os.path.join(DISP, "data", "superset_mixed.pt"), weights_only=False)
    for pre in sys.argv[1:]:
        rows = []
        for sp in (0, 1, 2):
            files = sorted(glob.glob(f"{pre}_s*_sp{sp}.pt"))
            gs = [g for g in G if sp in g.split_test]
            if not files or not gs:
                continue
            runs = [run(torch.load(f, weights_only=False, map_location="cpu"), gs, device) for f in files]
            items = []
            for i, g in enumerate(gs):
                pr = torch.stack([r[i]["pred"] for r in runs]).mean(0)
                cl = [r[i]["cell"] for r in runs if r[i]["cell"] is not None]
                eh = [r[i]["e_head"] for r in runs if r[i]["e_head"] is not None]
                items.append(dict(pred=pr, y=g.disp, mask=torch.ones(g.num_nodes, dtype=torch.bool), dist=g.dist,
                                  unstable=bool(g.unstable), null=False, folder=g.folder, key=g.kind,
                                  cell_pred=torch.stack(cl).mean(0) if cl else torch.zeros(3), has_cell=bool(cl),
                                  cell_y=g.cell_change_frac.view(3) * g.cell0[0].norm(dim=-1),
                                  e_harm=float(np.mean([r[i]["e_harm"] for r in runs])),
                                  e_head=float(np.mean(eh)) if eh else np.nan, e_true=float((g.pe1 - g.pe0).sum())))
            pd = [it for it in items if it["key"] != "none"]; no = [it for it in items if it["key"] == "none"]
            st = [it for it in items if not it["unstable"]]
            et = np.array([it["e_true"] for it in st]); big = np.abs(et) > 1e-3
            erel = lambda k: float(np.median(np.abs(np.array([it[k] for it in st]) - et)[big] / np.abs(et[big])))  # noqa: E731
            cm = [float((it["cell_pred"] - it["cell_y"]).abs().mean()) for it in st if it["has_cell"]]
            rows.append([metrics(items)["r_disp"], metrics(pd, R_EVAL_POINT)["r_disp"], metrics(no)["r_disp"],
                         np.mean(cm) if cm else np.nan, erel("e_harm"), erel("e_head"), len(items)])
        if not rows:
            print(pre, "no files"); continue
        r = np.array(rows, dtype=float).mean(0)
        print(f"{os.path.basename(pre)[:90]}\n   mixed r_disp all {r[0]:.3f} | point region {r[1]:.3f} | pure faults {r[2]:.3f} | "
              f"c MAE {r[3]:.4f} A | E rel err: -1/2F0.d {r[4]:.3f} head {r[5]:.3f} | splits {len(rows)} cells/split {r[6]:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
