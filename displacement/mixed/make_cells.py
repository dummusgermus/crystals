"""Mixed-defect test cells, stage 0 (training venv): point defect INSIDE a planar-fault cell.

Takes the unrelaxed planar-fault cells (Laves_Planar_Defects, the same inputs as the planar training data) of
every compound (split like the other data by train_disp's union compound split, so cells of training compounds
are mixed TRAINING data and cells of test compounds the mixed test set), repeats them in-plane (a, b >= --min-ab A) and creates per cell
  kind 'none'   the pure fault supercell (reference: the model must reproduce its planar behaviour)
  kind 'near'   one antisite (A <-> B type swap) on a fault-layer atom
  kind 'far'    one antisite on the atom farthest from the fault layer
  kind 'rand'   one antisite on a random atom
Fault-layer atoms = is_defect labels of the planar training data. Antisites as in the point-defect data.

    python displacement/mixed/make_cells.py <out.pkl> [--per-compound 2] [--min-ab 15] [--max-atoms 450]
Output: list of dicts (numpy): id, compound, folder, kind, split_test (list of split seeds with the compound
in test), elements (A, B), types (1/2), pos, cell (rows a, b, c), is_fault, point_idx (-1 for 'none').
Stage 1: grace_relax.py (GRACE venv), stage 2: build_mixed.py.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE); ROOT = os.path.dirname(DISP)
sys.path.insert(0, DISP)
from build_superset import old_planar_labels  # noqa: E402
from dumpio import read_dump  # noqa: E402


def test_compounds():
    """Compound -> list of union-split seeds in which it is a TEST compound (as train_disp.split_by_compound)."""
    keys = set()
    for d in ("point", "planar"):
        keys |= {f"{g.element_a}-{g.element_b}" for g in torch.load(os.path.join(DISP, "data", f"superset_{d}.pt"),
                                                                     weights_only=False)}
    out = {}
    for seed in (0, 1, 2):
        k = sorted(keys); rng = np.random.default_rng(1000 + seed); rng.shuffle(k)
        n2 = int(round(0.7 * len(k))) + int(round(0.15 * len(k)))
        for c in k[n2:]:
            out.setdefault(c, []).append(seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--per-compound", type=int, default=2)
    ap.add_argument("--min-ab", type=float, default=15.0); ap.add_argument("--max-atoms", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tc = test_compounds()
    planar = torch.load(os.path.join(DISP, "data", "superset_planar.pt"), weights_only=False)
    ok = {g.folder for g in planar if not bool(g.unstable) and bool((g.is_defect > 0.5).any())}
    labels = old_planar_labels()
    by_c = {}
    for f in sorted(ok):
        c = f.split("_", 1)[0]
        by_c.setdefault(c, []).append(f)
    out = []
    for c in sorted(by_c):
        fs = list(by_c[c]); rng.shuffle(fs); n_done = 0
        for f in fs:
            if n_done >= a.per_compound:
                break
            u = read_dump(os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS", f, "minimised.dump"))
            H = u["cell"]; pos = u["pos"] - u["origin"]
            na, nb = (int(np.ceil(a.min_ab / np.linalg.norm(H[k]))) for k in (0, 1))
            if len(pos) * na * nb > a.max_atoms:
                continue
            lab = labels[f]
            isf = np.array([lab.get(int(i), 0.0) for i in u["ids"]]) > 0.5
            shifts = np.array([i * H[0] + j * H[1] for i in range(na) for j in range(nb)])
            P = (pos[None] + shifts[:, None]).reshape(-1, 3)
            T = np.tile(u["type"].astype(int), len(shifts)); F = np.tile(isf, len(shifts))
            C = np.array([na * H[0], nb * H[1], H[2]])
            # distance of every atom to the nearest fault atom (min image), for the 'far' site
            inv = np.linalg.inv(C); fa = np.flatnonzero(F)
            dfr = (P[:, None] - P[None, fa]) @ inv; dfr -= np.round(dfr)
            dist_f = np.linalg.norm(dfr @ C, axis=-1).min(1)
            A, B = c.split("-")
            base = dict(compound=c, folder=f, split_test=tc.get(c, []), elements=(A, B), pos=P, cell=C, is_fault=F,
                        reps=(na, nb))
            near = int(rng.choice(fa)); far = int(rng.choice(np.flatnonzero(dist_f >= dist_f.max() - 0.5)))
            rnd = int(rng.integers(len(P)))
            for kind, pi in (("none", -1), ("near", near), ("far", far), ("rand", rnd)):
                t = T.copy()
                if pi >= 0:
                    t[pi] = 3 - t[pi]  # antisite: A <-> B
                out.append(dict(base, id=f"{f}|{kind}", kind=kind, types=t, point_idx=pi,
                                dist_fault_of_point=float(dist_f[pi]) if pi >= 0 else np.nan))
            n_done += 1
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    n = np.array([len(o["pos"]) for o in out])
    print(f"{len(out)} cells ({len(by_c)} compounds, {sum(bool(o['split_test']) for o in out)} cells of test compounds) atoms {n.min()}-{n.max()} (median "
          f"{int(np.median(n))}) -> {a.out}")


if __name__ == "__main__":
    main()
