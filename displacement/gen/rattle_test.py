"""Symmetry-breaking test for the pilot cells where our GRACE relaxation and the team's LAMMPS labels disagree.

For every pilot defect cell with displacement rel. diff > 5% (8.9 A region; ref_dE is a noisy dump sum):
  ours+rattle   their unrelaxed cell + random displacements (sigma 0.02 A, seeds 0,1,2), relaxed (FIRE+LBFGS, fmax 1e-4)
  ref+rattle    their LAMMPS-relaxed cell + the same rattle, relaxed again: if it drops to a lower energy, the
                LAMMPS state was a saddle point (symmetry-constrained), not a minimum
Energies relative to their unrelaxed cell (GRACE e0), comparable to ref_dE.
    python -I displacement/gen/rattle_test.py <out.pkl> --part k --nparts n     (GRACE venv)
    python -I displacement/gen/rattle_test.py report 'displacement/gen_data/rattle_part*.pkl'
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_pilot import optimise, wrap_rel  # noqa: E402

SIGMA, SEEDS, FMAX = 0.02, (0, 1, 2), 1e-4
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), "gen_data")


def disagreeing():
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "pilot_part*.pkl"))):
        for d in pickle.load(open(p, "rb"))["defect"]:
            rel = wrap_rel(d)
            if np.isfinite(rel) and rel > 0.05:
                out.append(d)
    return out


def relax_from(calc, d, pos, seed):
    from ase import Atoms
    A, B = d["folder"].split("_", 1)[1].split("-")
    at = Atoms(symbols=d["symbols"], positions=pos, cell=d["cell"], pbc=True) if "symbols" in d else None
    if at is None:
        raise KeyError("symbols")
    if seed is not None:
        at.positions += np.random.default_rng(seed).normal(0, SIGMA, at.positions.shape)
    at.calc = calc
    t = time.time(); conv, n = optimise(at, FMAX, 3000)
    return dict(e=at.get_potential_energy(), pos1=at.positions.copy(), conv=conv, steps=n, sec=time.time() - t)


def run(a):
    from tensorpotential.calculator import TPCalculator
    from point_pilot import POT, cell_symbols
    calc = TPCalculator(POT)
    cells = disagreeing()[a.part:: a.nparts]
    print(f"{len(cells)} cells in this part", flush=True)
    res = []
    for d in cells:
        d = dict(d); d["symbols"] = cell_symbols(d)
        r = dict(folder=d["folder"], key=d["key"], dE=d["dE"], ref_dE=d["ref_dE"], e0=d["e0"], runs=[])
        for kind, pos in (("ours+rattle", d["pos0"]), ("ref+rattle", d["ref_pos1"])):
            for s in SEEDS if kind == "ours+rattle" else (0, None):
                x = relax_from(calc, d, pos, s); x.update(kind=kind if s is not None else "ref (no rattle)", seed=s)
                x["dE"] = x["e"] - d["e0"]; del x["e"]
                r["runs"].append(x)
                print(f"{d['folder']} {d['key']} {x['kind']} seed {s}: dE {x['dE']:.4f} (ours {d['dE']:.4f}, LAMMPS "
                      f"{d['ref_dE']:.4f}) conv {x['conv']} {x['sec']:.0f}s", flush=True)
        res.append(r)
        pickle.dump(res, open(a.out, "wb"))
    print("done", flush=True)


def report(a):
    R = [r for p in sorted(glob.glob(a.out)) for r in pickle.load(open(p, "rb"))]
    print(f"{'cell':32s} {'LAMMPS':>8s} {'ours':>8s} {'ours+rattle (3)':>26s} {'ref->GRACE':>10s} {'ref+rattle':>10s}")
    for r in R:
        g = lambda k: [x["dE"] for x in r["runs"] if x["kind"] == k]  # noqa: E731
        print(f"{r['folder'] + ' ' + r['key']:32s} {r['ref_dE']:8.4f} {r['dE']:8.4f} "
              f"{' '.join(f'{v:8.4f}' for v in g('ours+rattle')):>26s} {g('ref (no rattle)')[0]:10.4f} {g('ref+rattle')[0]:10.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("out", nargs="?")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    if a.what == "report":
        report(a)
    else:
        a.out = a.what; run(a)
