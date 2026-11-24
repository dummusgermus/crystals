"""Production model: unified point+planar+mixed ensemble (PaiNN multi-task with harmonic-residual energy x2 + DefiNet x2).

    python displacement/predict_disp.py <superset.pt> <out.pt> [--ckpt-dir displacement/production]

Input: superset-format graphs (displacement/build_superset.py or mixed/build_mixed.py: unrelaxed structure +
GRACE single point; graph.domain = 'point', 'planar' or 'mixed' (point defect in a fault cell, with is_fault /
dist_fault); 'mixed' needs unified checkpoints, train_disp.py --unified). Relaxed targets are not needed (missing disp / pe1 are zero-filled).
Every *.pt checkpoint in --ckpt-dir (written by train_disp.py --final --save-preds) is one ensemble member;
the members' displacements (and planar c-vector changes) are averaged.

Output (list of dicts, one per input graph, same order):
  folder, key, domain, node_ids (superset node indices of the view), disp (N,3) predicted displacement (A,
  planar: non-affine part, zero mean), rel_relaxed = rel + disp, cell_y (planar: c-vector change, A, along
  initial a, b, c; None for point), members,
  e_harm  relaxation energy (eV) from the ensemble displacement, -1/2 sum F0 . d over the view (harmonic estimate),
  e_head  relaxation energy (eV) from the learned energy head, mean over members that have one (else None),
  e_relax recommended estimate: e_head if the members learned the residual over -1/2 F0.d (--e-harm) or for
          point defects; e_harm otherwise (runs_energy study).
Point-defect predictions are meaningful within 8.9 A of the defect (the training loss region).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from atom_encodings import encode_dataset  # noqa: E402
from edge_features import encode_edges  # noqa: E402
from models import PaiNNDisplacement  # noqa: E402
from ext.wrapper import ExtRelax  # noqa: E402
from train_disp import atom_energy, joint_gauge, unify  # noqa: E402
from views import make_view  # noqa: E402


def build(ck, device):
    c = ck["config"]
    if c["model"] == "painn":
        m = PaiNNDisplacement(ck["x_dim"], ck["e_dim"], hidden=c["hidden"], num_layers=c["layers"],
                              force_init=c["head"] == "eqvF", cell_head=True, norm=c["norm"], zemb=c["zemb"],
                              energy_head=c.get("energy_weight", 0) > 0, e_coupling=c.get("e_coupling", False))
    else:
        m = ExtRelax(c["model"], feat_dim=ck["x_dim"] if c["ext_feats"] else None, force_init=c["ext_force"],
                     hidden=c["hidden"], num_layers=c["layers"], planar=False, mode="fixed")
        m.set_scales(1.0, ck["cell_scale"])
    m.load_state_dict(ck["state"]); return m.to(device).eval()


def views(graphs, ck):
    c = ck["config"]
    out = []
    for g in graphs:
        g = g.clone()
        for k, shape in (("disp", (g.num_nodes, 3)), ("pe1", (g.num_nodes,))):
            if getattr(g, k, None) is None:
                setattr(g, k, torch.zeros(shape))
        if g.domain != "point" and getattr(g, "cell_change_frac", None) is None:
            g.cell_change_frac = torch.zeros(1, 3)
        if getattr(g, "unstable", None) is None:
            g.unstable = False
        v = make_view(g, c["spec"]); v.dom = torch.tensor([{"point": 0, "planar": 1, "mixed": 2}[g.domain]])
        v.node_ids = torch.nonzero(g.dist <= c["spec"]["rt"] + c["spec"]["buf"] + 1e-6).view(-1) \
            if g.domain == "point" and c["spec"]["rt"] > 0 and not getattr(g, "full_cell", False) \
            else torch.arange(g.num_nodes)
        out.append(v)
    if c.get("unified"):
        out = unify(out)
    elif any(g.domain == "mixed" for g in graphs):
        raise SystemExit("mixed cells need unified checkpoints (train_disp.py --unified)")
    out = encode_dataset(out, c["encoding"])
    if not c.get("no_domflag"):
        for v in out:
            v.x = torch.cat([v.x, torch.nn.functional.one_hot(v.dom.expand(v.num_nodes), 2).float()], -1)
    out = encode_edges(out, c["edge_enc"])
    n = ck["norm"]
    for v in out:
        if n is not None:
            v.x = (v.x - n["xm"]) / n["xs"]; v.edge_attr = (v.edge_attr - n["em"]) / n["es"]
        v.dscale = torch.tensor([ck["dscales"][min(int(v.dom), len(ck["dscales"]) - 1)]])
        if ck.get("escales"):
            v.edsc = torch.tensor([ck.get("e_unit") or ck["escales"][int(v.dom)][0]])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "production"))
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = torch.load(a.inp, weights_only=False)
    cks = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")))
    if not cks:
        raise SystemExit(f"no checkpoints in {a.ckpt_dir}")
    acc = None; eh = False
    for p in cks:
        ck = torch.load(p, weights_only=False, map_location="cpu")
        model, vs = build(ck, device), views(graphs, ck)
        eh = eh or bool(ck["config"].get("e_harm"))
        disp, cells, ens = [], [], []
        for b in DataLoader(vs, batch_size=a.bs):
            b = b.to(device)
            d, cell = model(b)
            dg = joint_gauge(model, b, d); d = dg * b.dscale[b.batch].unsqueeze(-1)
            ea = atom_energy(model, b, dg, ck["config"].get("e_harm", False), False) * b.edsc[b.batch] \
                if getattr(model, "e_atom", None) is not None else None
            for k in range(b.num_graphs):
                disp.append(d[b.batch == k].cpu())
                cells.append((cell[k] * ck["cell_scale"]).cpu() if cell is not None and int(b.dom[k]) >= 1 else None)
                ens.append(float(ea[b.batch == k].sum()) if ea is not None else None)
        if acc is None:
            acc = dict(disp=[[x] for x in disp], cell=[[c] if c is not None else [] for c in cells],
                       e=[[e] if e is not None else [] for e in ens], vs=vs)
        else:
            for i, (x, c, e) in enumerate(zip(disp, cells, ens)):
                acc["disp"][i].append(x)
                if c is not None:
                    acc["cell"][i].append(c)
                if e is not None:
                    acc["e"][i].append(e)
        print(f"{os.path.basename(p)}: {len(vs)} graphs", flush=True)
    res = []
    for i, g in enumerate(graphs):
        v = acc["vs"][i]; d = torch.stack(acc["disp"][i]).mean(0)
        e_harm = float(-0.5 * (v.force0.double() * d.double()).sum())
        e_head = sum(acc["e"][i]) / len(acc["e"][i]) if acc["e"][i] else None
        res.append(dict(e_harm=e_harm, e_head=e_head, e_relax=e_head if (e_head is not None and (eh or g.domain == "point")) else e_harm,
                        folder=g.folder, key=v.key, domain=g.domain, node_ids=v.node_ids, disp=d,
                        rel_relaxed=g.rel[v.node_ids] + d,
                        cell_y=torch.stack(acc["cell"][i]).mean(0) if acc["cell"][i] else None, members=len(cks)))
    torch.save(res, a.out)
    print(f"-> {a.out}: {len(res)} structures, {len(cks)} members")


if __name__ == "__main__":
    main()
