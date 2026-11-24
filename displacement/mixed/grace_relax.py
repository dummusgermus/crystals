"""Mixed-defect test cells, stage 1 (GRACE venv): GRACE single point + relaxation, the planar-data protocol.

LAMMPS protocol of the planar training data (Laves_Screen fire_cg.in): GRACE-2L-UEA-OMAT-large,
fix box/relax z 0 yz 0 xz 0 (only the c vector relaxes), cg + fire, ftol 1e-6, 3 rounds. Here: ASE
FrechetCellFilter with mask (zz, yz, xz), FIRE to fmax --fmax (default 1e-3 eV/A), then LBFGS if needed.

    python -I displacement/mixed/grace_relax.py cells <cells.pkl> <out.pkl> [--part k --nparts n]
    python -I displacement/mixed/grace_relax.py validate <out.pkl> [--n 24]
'validate' relaxes --n pure planar training cells (Laves_Planar_Defects inputs) with this protocol and stores
the LAMMPS references (Laves_Screen) next to them, to check that the ASE protocol reproduces the labels.
Output per structure: pe0, force0 (unrelaxed), pos1, cell1, pe1 (relaxed), conv, steps, fmax.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE); CODE = os.path.dirname(DISP)
POT = os.path.join(CODE, "Laves_Screen", "STATIC_FILES", "POTENTIALS", "GRACE-2L-UEA-OMAT-large")
sys.path.insert(0, DISP)
from dumpio import read_dump  # noqa: E402


def relax(calc, C, symbols, pos, cell, fmax, steps):
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS
    at = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True); at.calc = calc
    f0 = at.get_forces().copy(); pe0 = np.asarray(calc.results[C.PREDICT_ATOMIC_ENERGY]).ravel()[: len(at)].copy()
    filt = FrechetCellFilter(at, mask=[0, 0, 1, 1, 1, 0])  # voigt xx yy zz yz xz xy: c vector only
    n = 0; conv = False
    for opt_cls, k in ((FIRE, steps), (LBFGS, steps // 3)):
        opt = opt_cls(filt, logfile=None)
        conv = bool(opt.run(fmax=fmax, steps=k)); n += opt.nsteps
        if conv:
            break
    at.get_potential_energy()
    pe1 = np.asarray(calc.results[C.PREDICT_ATOMIC_ENERGY]).ravel()[: len(at)].copy()
    fm = float(np.linalg.norm(at.get_forces(), axis=1).max())
    return dict(pe0=pe0, force0=f0, pos1=at.positions.copy(), cell1=np.array(at.cell), pe1=pe1, conv=conv,
                steps=n, fmax=fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cells", "validate"]); ap.add_argument("paths", nargs="+")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--fmax", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    from tensorpotential.calculator import TPCalculator
    from tensorpotential import constants as C
    calc = TPCalculator(POT, extra_properties=[C.PREDICT_ATOMIC_ENERGY], truncate_extras_by_natoms=True)
    t0 = time.time(); out = []
    if a.what == "cells":
        cells = pickle.load(open(a.paths[0], "rb"))[a.part:: a.nparts]
        dst = a.paths[1]
        for k, c in enumerate(cells):
            sym = [c["elements"][t - 1] for t in c["types"]]
            r = relax(calc, C, sym, c["pos"], c["cell"], a.fmax, a.steps)
            out.append(dict(id=c["id"], **r))
            print(f"{k + 1}/{len(cells)} {c['id']} n={len(sym)} conv={r['conv']} steps={r['steps']} "
                  f"fmax={r['fmax']:.1e} dE={float((r['pe1'] - r['pe0']).sum()):.4f} | {time.time() - t0:.0f}s", flush=True)
    else:
        dst = a.paths[0]
        root_u = os.path.join(CODE, "Laves_Planar_Defects", "SIMULATIONS"); root_r = os.path.join(CODE, "Laves_Screen", "SIMULATIONS")
        fs = sorted(f for f in os.listdir(root_u) if os.path.isfile(os.path.join(root_r, f, "minimised.dump")))
        rng = np.random.default_rng(0); fs = list(rng.choice(fs, a.n, replace=False))
        for k, f in enumerate(fs):
            u = read_dump(os.path.join(root_u, f, "minimised.dump")); rr = read_dump(os.path.join(root_r, f, "minimised.dump"))
            els = f.split("_", 1)[0].split("-")
            sym = [els[int(t) - 1] for t in u["type"]]
            try:
                r = relax(calc, C, sym, u["pos"] - u["origin"], u["cell"], a.fmax, a.steps)
            except Exception as e:  # noqa: BLE001
                print("error", f, e, flush=True); continue
            out.append(dict(id=f, ref_pos0=u["pos"] - u["origin"], ref_cell0=u["cell"], ref_pe0=u["pe"], ref_force0=u["force"],
                            ref_pos1=rr["pos"] - rr["origin"], ref_cell1=rr["cell"], ref_pe1=rr["pe"], **r))
            print(f"{k + 1}/{len(fs)} {f} conv={r['conv']} steps={r['steps']} | {time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(dst, "wb"), protocol=4)
    print(f"done {len(out)} -> {dst} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
