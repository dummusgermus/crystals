"""Build "superset" displacement graphs for point defects and planar faults.

One rich graph per structure; every dataset variant used in the displacement study
is a cheap *view* of it (displacement/views.py): which nodes, which edges, which
features. Stored per graph:

nodes   type (1=A, 2=B), z, pe0 (initial per-atom PE), pe1 (relaxed), force0 (initial
        force, eV/A), rel (position relative to the defect / cell origin, A),
        dvec (min-image vector from the nearest defect atom to the atom; zeros if none),
        dist (|dvec|), is_defect, c3_k2/c4_k2/c3_k3/c4_k3 (3-/4-cycles through the
        atom in the 2- / 3-shell neighbour graph, computed in the full periodic crystal),
        disp (TARGET: displacement during relaxation, A; for planar the non-affine
        part, i.e. fractional-coordinate change mapped with the initial cell, mean removed)
edges   all pairs within R_EDGE (periodic images included, both directions):
        edge_index (src -> dst), edge_vec (r_dst - r_src), edge_dist, edge_shell
        (neighbour-shell rank of dst seen from src, 1 = nearest)
graph   domain, folder, element_a/b, defect key or stacking sequence, cell0,
        cell_change_frac (planar: change of the c vector in units of the initial
        a, b, c vectors), dE_total, unstable flag, n_full

Point defects: nodes = atoms within R_NODE of the defect (the full 3000-atom cell
is used for shells and cycles). Vacancies are skipped (as in the PE pipeline).

    python displacement/build_superset.py point|planar [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch_geometric.data import Data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from dumpio import read_dump  # noqa: E402

# same element order as graph_maker.ELEMENTS (Z = 1..100); not imported because
# graph_maker pulls in ovito, which needs libOpenGL (missing on the compute nodes)
_ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
             "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
             "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th "
             "Pa U Np Pu Am Cm Bk Cf Es Fm").split()
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_ELEMENTS)}

R_EDGE = 6.0
R_NODE_POINT = 14.0
SHELL_TOL_REL = 0.02
EXCLUDED = {"Am", "Bk", "Cm"}  # not in element_properties.json


# ----------------------------------------------------------------------------- geometry
def neighbour_list(pos, cell, centers, r_cut):
    """All (i in centers, j any atom, image) pairs with 0 < |r_j + n.cell - r_i| <= r_cut.

    cell rows are lattice vectors. Returns src (index into centers' atom ids), dst atom
    index, vectors."""
    inv = np.linalg.inv(cell)
    # perpendicular widths -> number of images needed per axis
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    nimg = [int(np.ceil(r_cut / w + 0.5)) for w in widths]
    # wrap fractional coordinates so that min-image + images is exact
    frac = (pos @ inv)
    imgs = np.array([(a, b, c) for a in range(-nimg[0], nimg[0] + 1) for b in range(-nimg[1], nimg[1] + 1)
                     for c in range(-nimg[2], nimg[2] + 1)], dtype=float)
    shifts = imgs @ cell
    src_l, dst_l, vec_l = [], [], []
    for ci in centers:
        d0 = frac - frac[ci]
        d0 -= np.round(d0)  # min image in fractional space
        base = d0 @ cell  # (N,3)
        # only atoms within r_cut + max image shift can matter; test all images
        v = base[None, :, :] + shifts[:, None, :]  # (I,N,3)
        dist = np.linalg.norm(v, axis=-1)
        m = (dist <= r_cut) & (dist > 1e-6)
        ii, jj = np.nonzero(m)
        src_l.append(np.full(len(jj), ci)); dst_l.append(jj); vec_l.append(v[ii, jj])
    return np.concatenate(src_l), np.concatenate(dst_l), np.concatenate(vec_l)


def shell_ranks(src, dist):
    """Neighbour-shell rank of each pair as seen from its source atom."""
    rank = np.zeros(len(src), dtype=np.int64)
    order = np.lexsort((dist, src))
    s_sorted = src[order]; d_sorted = dist[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for seg in np.split(np.arange(len(order)), bounds):
        d = d_sorted[seg]
        tol = max(d[0] * SHELL_TOL_REL, 1e-6)
        r = np.ones(len(d), dtype=np.int64)
        cur = 1; ref = d[0]
        for k in range(1, len(d)):
            if d[k] - ref > tol:
                cur += 1; ref = d[k]
            r[k] = cur
        rank[order[seg]] = r
    return rank


def cycle_counts(pos, cell, centers, thr):
    """3- and 4-cycles through each center atom in the graph where atoms p, q are bonded if
    |r_pq| <= thr[p] or <= thr[q] (thr = k-shell threshold per atom). Computed on an
    explicit real-space cluster, so periodic images are handled correctly."""
    rc = 2 * thr.max() + 0.05
    src, dst, vec = neighbour_list(pos, cell, centers, rc)
    out = np.zeros((len(centers), 2))
    cidx = {c: k for k, c in enumerate(centers)}
    order = np.argsort(src, kind="stable")
    src, dst, vec = src[order], dst[order], vec[order]
    bounds = np.flatnonzero(np.diff(src)) + 1
    for seg in np.split(np.arange(len(src)), bounds):
        c = src[seg[0]]
        P = np.vstack([np.zeros(3), vec[seg]])
        T = np.concatenate([[thr[c]], thr[dst[seg]]])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        A = ((D <= T[:, None] + 1e-6) | (D <= T[None, :] + 1e-6)) & (D > 1e-6)
        A = A.astype(float)
        A2 = A @ A
        deg = A.sum(1)
        c3 = (A2 @ A)[0, 0] / 2
        a4 = (A2 @ A2)[0, 0]
        c4 = (a4 - deg[0] ** 2 - (A[0] * (deg - 1)).sum()) / 2
        out[cidx[c]] = (c3, c4)
    return out


def shell_thresholds(pos, cell, atoms, ks=(2, 3)):
    src, dst, vec = neighbour_list(pos, cell, atoms, 5.0)
    dist = np.linalg.norm(vec, axis=1)
    rank = shell_ranks(src, dist)
    thr = {}
    for k in ks:
        t = np.zeros(len(pos))
        for a in atoms:
            m = (src == a) & (rank <= k)
            t[a] = dist[m].max()
        thr[k] = t
    return thr


def nearest_defect_vec(pos, cell, defect_idx):
    """Min-image vector from the nearest defect atom to every atom (zeros if no defect)."""
    if len(defect_idx) == 0:
        return np.zeros_like(pos)
    inv = np.linalg.inv(cell)
    best = np.full(len(pos), np.inf); out = np.zeros_like(pos)
    imgs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) @ cell
    for d in defect_idx:
        f = (pos - pos[d]) @ inv; f -= np.round(f)
        v = (f @ cell)[None] + imgs[:, None]
        n = np.linalg.norm(v, axis=-1); k = n.argmin(0)
        vv = v[k, np.arange(len(pos))]; nn = n[k, np.arange(len(pos))]
        upd = nn < best; best[upd] = nn[upd]; out[upd] = vv[upd]
    return out


# ----------------------------------------------------------------------------- graph assembly
def assemble(nodes_idx, pos, cell, types, pe0, pe1, force0, disp, dvec, is_def, z_of_type, meta,
             r_edge=None, max_shell=None, cycles=True):
    nodes_idx = np.asarray(nodes_idx)
    remap = -np.ones(len(pos), dtype=np.int64); remap[nodes_idx] = np.arange(len(nodes_idx))
    src, dst, vec = neighbour_list(pos, cell, nodes_idx, r_edge or R_EDGE)
    dist = np.linalg.norm(vec, axis=1)
    rank = shell_ranks(src, dist)  # computed with the FULL environment of src
    keep = remap[dst] >= 0
    if max_shell:
        keep &= rank <= max_shell
    src, dst, vec, dist, rank = src[keep], dst[keep], vec[keep], dist[keep], rank[keep]
    if cycles:
        thr = shell_thresholds(pos, cell, np.unique(np.concatenate([nodes_idx, neighbour_list(pos, cell, nodes_idx, 7.0)[1]])))
        cyc2 = cycle_counts(pos, cell, nodes_idx, thr[2])
        cyc3 = cycle_counts(pos, cell, nodes_idx, thr[3])
    else:
        cyc2 = cyc3 = np.full((len(nodes_idx), 2), np.nan)
    t = lambda a, dt=torch.float: torch.tensor(np.asarray(a), dtype=dt)  # noqa: E731
    g = Data(
        num_nodes=len(nodes_idx),
        type=t(types[nodes_idx], torch.long),
        z=t([z_of_type[int(v)] for v in types[nodes_idx]], torch.long),
        pe0=t(pe0[nodes_idx]), pe1=t(pe1[nodes_idx]), force0=t(force0[nodes_idx]),
        dvec=t(dvec[nodes_idx]), dist=t(np.linalg.norm(dvec[nodes_idx], axis=1)),
        is_defect=t(is_def[nodes_idx]), disp=t(disp[nodes_idx]),
        cyc_k2=t(cyc2), cyc_k3=t(cyc3),
        edge_index=t(np.stack([remap[src], remap[dst]]), torch.int32 if max_shell else torch.long),
        edge_vec=t(vec), edge_dist=t(dist), edge_shell=t(rank, torch.int8 if max_shell else torch.long),
        cell0=t(cell).unsqueeze(0),
    )
    for k, v in meta.items():
        setattr(g, k, v)
    return g


# ----------------------------------------------------------------------------- point
def read_data_types(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Atoms"))
    out = {}
    for l in lines[i + 2:]:
        p = l.split()
        if not p:
            break
        out[int(p[0])] = int(p[1])
    return out


FULL = False


def build_point_one(f):
    folder = os.path.basename(os.path.dirname(f))
    a, b = folder.split("_", 1)[1].split("-")
    key = os.path.basename(f)[len("unrelaxed_"):-len(".dump")]
    did = int(key.split("-")[0])
    u = read_dump(f); r = read_dump(f.replace("unrelaxed_", "relaxed_"))
    assert (u["ids"] == r["ids"]).all()
    tmap = read_data_types(f.replace(".dump", ".data"))
    types = np.array([tmap[int(i)] for i in u["ids"]])
    cell = u["cell"]; inv = np.linalg.inv(cell)
    d = r["pos"] - u["pos"]; fr = d @ inv; d = (fr - np.round(fr)) @ cell
    i0 = int(np.where(u["ids"] == did)[0][0])
    fr = (u["pos"] - u["pos"][i0]) @ inv; rel = (fr - np.round(fr)) @ cell
    dist = np.linalg.norm(rel, axis=1)
    nodes = np.flatnonzero(dist <= (np.inf if FULL else R_NODE_POINT))
    nodes = nodes[np.argsort(dist[nodes], kind="stable")]  # defect first
    is_def = np.zeros(len(types)); is_def[i0] = 1.0
    far = np.linalg.norm(d, axis=1)[dist > 12]
    meta = dict(domain="point", folder=folder, element_a=a, element_b=b, defect_key=key,
                dE_total=float(r["pe"].sum() - u["pe"].sum()), n_full=len(types),
                far_disp_median=float(np.median(far)), unstable=bool(np.median(far) > 0.05),
                particle_ids=torch.tensor(u["ids"][nodes]))
    kw = dict(r_edge=5.8, max_shell=4, cycles=False) if FULL else {}
    g = assemble(nodes, u["pos"], cell, types, u["pe"], r["pe"], u["force"], d, rel, is_def,
                 {1: SYMBOL_TO_Z[a], 2: SYMBOL_TO_Z[b]}, meta, **kw)
    g.full_cell = FULL
    g.rel = torch.tensor(rel[nodes], dtype=torch.float)
    return g


def point_files():
    out = []
    root = os.environ.get("DISP_SIM_ROOT", ROOT)  # generated data: displacement/gen/point_gen.py output dir
    for f in sorted(glob.glob(os.path.join(root, "SIMULATIONS", "C15_*", "unrelaxed_*.dump"))):
        folder = os.path.basename(os.path.dirname(f))
        if folder.endswith("_MIN") or "--1-" in os.path.basename(f):
            continue
        if set(folder.split("_", 1)[1].split("-")) & EXCLUDED:
            continue
        if not os.path.isfile(f.replace("unrelaxed_", "relaxed_")):
            continue
        out.append(f)
    return out


# ----------------------------------------------------------------------------- planar
_OLD_PLANAR = None


def old_planar_labels():
    """is_defect labels from the existing planar PE dataset, keyed by (folder, particle id)."""
    global _OLD_PLANAR
    if _OLD_PLANAR is None:
        ds = torch.load(os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt"), weights_only=False)
        _OLD_PLANAR = {g.folder: dict(zip(g.particle_ids.tolist(), g.x[:, 2].tolist())) for g in ds}
    return _OLD_PLANAR


def build_planar_one(args):
    folder, labels = args
    a, b = folder.split("_", 1)[0].split("-")
    seq = folder.split("_", 1)[1]
    u = read_dump(os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS", folder, "minimised.dump"))
    r = read_dump(os.path.join(ROOT, "Laves_Screen", "SIMULATIONS", folder, "minimised.dump"))
    assert (u["ids"] == r["ids"]).all()
    H0, H1 = u["cell"], r["cell"]
    s0 = (u["pos"] - u["origin"]) @ np.linalg.inv(H0)
    s1 = (r["pos"] - r["origin"]) @ np.linalg.inv(H1)
    ds = s1 - s0; ds -= np.round(ds)
    d = ds @ H0; d -= d.mean(0)
    dc = (H1[2] - H0[2]) @ np.linalg.inv(H0)  # c-vector change in units of a, b, c
    is_def = np.array([labels.get(int(i), 0.0) for i in u["ids"]])
    pos = u["pos"]
    dvec = nearest_defect_vec(pos, H0, np.flatnonzero(is_def > 0.5))
    meta = dict(domain="planar", folder=folder, element_a=a, element_b=b, stack_sequence=seq,
                dE_total=float(r["pe"].sum() - u["pe"].sum()), n_full=len(pos),
                cell_change_frac=torch.tensor(dc, dtype=torch.float).unsqueeze(0),
                max_disp=float(np.linalg.norm(d, axis=1).max()),
                particle_ids=torch.tensor(u["ids"]))
    meta["unstable"] = bool(meta["max_disp"] > 1.0 or abs(dc[2]) > 0.1)
    g = assemble(np.arange(len(pos)), pos, H0, u["type"], u["pe"], r["pe"], u["force"], d, dvec, is_def,
                 {1: SYMBOL_TO_Z[a], 2: SYMBOL_TO_Z[b]}, meta)
    g.rel = torch.tensor(pos - u["origin"], dtype=torch.float)
    return g


def planar_folders():
    lab = old_planar_labels()
    out = []
    for f in sorted(os.listdir(os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS"))):
        a, b = f.split("_", 1)[0].split("-")
        if {a, b} & EXCLUDED or f not in lab:
            continue
        if not os.path.isfile(os.path.join(ROOT, "Laves_Screen", "SIMULATIONS", f, "minimised.dump")):
            continue
        out.append((f, lab[f]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["point", "planar"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--full", action="store_true", help="point: all atoms of the periodic cell, shells <= 4, no cycles")
    a = ap.parse_args()
    global FULL
    FULL = a.full
    if a.domain == "point":
        items, fn = point_files(), build_point_one
    else:
        items, fn = planar_folders(), build_planar_one
    if a.limit:
        items = items[: a.limit]
    print(f"{a.domain}: {len(items)} structures", flush=True)
    graphs, errors = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for k, (it, fu) in enumerate(zip(items, futs)):
            try:
                graphs.append(fu.result())
            except Exception as e:  # noqa: BLE001
                errors.append((str(it)[:120], repr(e)))
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(items)}", flush=True)
    out = a.out or os.path.join(HERE, "data", f"superset_{a.domain}{'_full' if a.full else ''}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(graphs, out)
    n = np.array([g.num_nodes for g in graphs]); e = np.array([g.edge_index.shape[1] for g in graphs])
    summ = dict(domain=a.domain, n_graphs=len(graphs), n_errors=len(errors), errors=errors[:20],
                nodes_median=float(np.median(n)), edges_median=float(np.median(e)),
                n_unstable=int(sum(g.unstable for g in graphs)), r_edge=R_EDGE,
                r_node=R_NODE_POINT if a.domain == "point" else None)
    json.dump(summ, open(out.replace(".pt", "_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "errors"}), errors[:3], flush=True)


if __name__ == "__main__":
    main()
