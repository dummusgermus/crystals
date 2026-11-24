"""Pilot: can we generate the point-defect data ourselves with GRACE (ASE) and reproduce the team's LAMMPS labels?

Their pipeline (Pyroxene, SIMULATIONS/C15_<A-B>{_MIN,}/fire_cg.in, GRACE-2L-UEA-OMAT-large):
  bulk    MgCu2 C15 template, 5x5x5 conventional cells (3000 atoms), retyped A/B, box/relax aniso + cg, then fire,
          ftol 1e-8, 3 rounds -> basefile.data
  defect  one atom of basefile changed (antisite) or deleted (vacancy) on the 8a or 16d site, fixed cell,
          fire, ftol 1e-6 -> unrelaxed_* / relaxed_*.dump
Ours (ASE + TPCalculator, same potential file):
  bulk    C15 built from the space group (Fd-3m, A on 8a, B on 16d) at the template lattice constant, 24-atom
          conventional cell relaxed with FrechetCellFilter (normal strains only = aniso), FIRE then LBFGS
  defect  their unrelaxed defect cell, positions relaxed with FIRE then LBFGS, fixed cell
Compared: lattice constants and bulk energy/atom; per defect displacement (relaxed - unrelaxed) on atoms within
8.9 A of the defect (the training metric region) and on all atoms, and the relaxation energy.

    python -I displacement/gen/point_pilot.py relax <out.pkl> [--n-compounds 30 --part k --nparts n]   (GRACE venv)
    python -I displacement/gen/point_pilot.py compare 'displacement/gen_data/pilot_part*.pkl'
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE); CODE = os.path.dirname(DISP)
POT = os.path.join(CODE, "Laves_Screen", "STATIC_FILES", "POTENTIALS", "GRACE-2L-UEA-OMAT-large")
SIM = os.path.join(CODE, "SIMULATIONS")
sys.path.insert(0, DISP)
from dumpio import read_dump  # noqa: E402

A0_TEMPLATE = 34.8116836925 / 5  # MgCu2 template box / 5


def read_data(path):
    """LAMMPS data file -> (cell rows a,b,c (orthorhombic/triclinic), ids, types, positions)."""
    L = open(path).read().splitlines()
    lo = {}
    for l in L[:20]:
        p = l.split()
        if len(p) >= 4 and p[2] in ("xlo", "ylo", "zlo"):
            lo[p[2]] = (float(p[0]), float(p[1]))
        if len(p) >= 6 and p[3] == "xy":
            tilt = [float(v) for v in p[:3]]
    xy, xz, yz = tilt
    cell = np.array([[lo["xlo"][1] - lo["xlo"][0], 0, 0], [xy, lo["ylo"][1] - lo["ylo"][0], 0],
                     [xz, yz, lo["zlo"][1] - lo["zlo"][0]]])
    i = next(k for k, l in enumerate(L) if l.startswith("Atoms"))
    rows = []
    for l in L[i + 2:]:
        if not l.split():
            break
        rows.append([float(v) for v in l.split()[:5]])
    r = np.array(rows); o = np.argsort(r[:, 0])
    return cell, r[o, 0].astype(int), r[o, 1].astype(int), r[o, 2:5] - np.array([lo["xlo"][0], lo["ylo"][0], lo["zlo"][0]])


def lammps_final_energy(log):
    e = re.findall(r"Energy initial, next-to-last, final =\s*\n\s*(\S+)\s+(\S+)\s+(\S+)", open(log).read())
    return float(e[-1][2]) if e else np.nan


def compounds(n, seed=0):
    cs = sorted(d.split("_", 1)[1] for d in os.listdir(SIM)
                if d.startswith("C15_") and not d.endswith("_MIN") and glob.glob(os.path.join(SIM, d, "relaxed_*.dump"))
                and os.path.isfile(os.path.join(SIM, d + "_MIN", "bi_minimised.data")))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(cs, min(n, len(cs)), replace=False))


def optimise(at, fmax, steps, cell_mask=None):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS
    obj = FrechetCellFilter(at, mask=cell_mask) if cell_mask is not None else at
    n = 0; conv = False
    for cls, k in ((FIRE, steps), (LBFGS, steps // 3)):
        opt = cls(obj, logfile=None); conv = bool(opt.run(fmax=fmax, steps=k)); n += opt.nsteps
        if conv:
            break
    return conv, n


def bulk(calc, A, B, fmax):
    from ase.spacegroup import crystal
    at = crystal([A, B], basis=[(0, 0, 0), (5 / 8, 5 / 8, 5 / 8)], spacegroup=227, cellpar=[A0_TEMPLATE] * 3 + [90] * 3)
    assert len(at) == 24
    at.calc = calc
    conv, n = optimise(at, fmax, 3000, cell_mask=[1, 1, 1, 0, 0, 0])
    return dict(abc=np.array(at.cell.lengths()), e_atom=at.get_potential_energy() / len(at), conv=conv, steps=n)


def defect(calc, Cst, folder, f, fmax, rattle=0.02, seed=0):
    from ase import Atoms
    A, B = folder.split("_", 1)[1].split("-")
    key = os.path.basename(f)[len("unrelaxed_"):-len(".dump")]
    u = read_dump(f); r = read_dump(f.replace("unrelaxed_", "relaxed_"))
    _, ids, types, _ = read_data(f.replace(".dump", ".data"))
    tm = dict(zip(ids, types)); T = np.array([tm[int(i)] for i in u["ids"]])
    pos0 = u["pos"] - u["origin"]
    at = Atoms(symbols=[(A, B)[t - 1] for t in T], positions=pos0, cell=u["cell"], pbc=True); at.calc = calc
    _, bids, _, bpos = read_data(os.path.join(SIM, folder, "basefile.data"))
    site = bpos[int(np.where(bids == int(key.split("-")[0]))[0][0])]
    e0 = at.get_potential_energy()
    if rattle > 0:  # break the site symmetry, else antisites can stay on a symmetric saddle (gen/rattle_test.py)
        r = np.random.default_rng(seed).normal(0, rattle, at.positions.shape)
        at.positions += r - r.mean(0)  # zero net translation
    t = time.time(); conv, n = optimise(at, fmax, 3000); dt = time.time() - t
    e1 = at.get_potential_energy()
    return dict(folder=folder, key=key, site=site, ids=u["ids"], cell=u["cell"], pos0=pos0, pos1=at.positions.copy(),
                ref_pos1=r["pos"] - r["origin"], dE=e1 - e0, ref_dE=float(r["pe"].sum() - u["pe"].sum()),
                ref_e0=float(u["pe"].sum()), e0=e0, conv=conv, steps=n, sec=dt, natoms=len(at))


def run_relax(a):
    from tensorpotential.calculator import TPCalculator
    from tensorpotential import constants as Cst
    calc = TPCalculator(POT)
    cs = compounds(a.n_compounds)[a.part:: a.nparts]
    out = dict(bulk=[], defect=[]); t0 = time.time()
    for c in cs:
        A, B = c.split("-")
        b = bulk(calc, A, B, 1e-4)
        cell_ref, ids, _, _ = read_data(os.path.join(SIM, f"C15_{c}_MIN", "bi_minimised.data"))
        out["bulk"].append(dict(compound=c, **b, ref_abc=np.linalg.norm(cell_ref, axis=1) / 5,
                                ref_e_atom=lammps_final_energy(os.path.join(SIM, f"C15_{c}_MIN", "log.lammps")) / len(ids)))
        print(f"bulk {c}: a {b['abc'].round(4)} vs {out['bulk'][-1]['ref_abc'].round(4)} | {time.time() - t0:.0f}s", flush=True)
        for f in sorted(glob.glob(os.path.join(SIM, f"C15_{c}", "unrelaxed_*.dump"))):
            if not os.path.isfile(f.replace("unrelaxed_", "relaxed_")):
                continue
            try:
                d = defect(calc, Cst, f"C15_{c}", f, a.fmax, a.rattle)
            except Exception as e:  # noqa: BLE001
                print("error", f, e, flush=True); continue
            out["defect"].append(d)
            print(f"  {d['key']}: n={d['natoms']} conv={d['conv']} steps={d['steps']} {d['sec']:.0f}s dE {d['dE']:.4f} vs "
                  f"{d['ref_dE']:.4f} eV | {time.time() - t0:.0f}s", flush=True)
        pickle.dump(out, open(a.paths[0], "wb"), protocol=4)
    print(f"done {len(out['bulk'])} compounds, {len(out['defect'])} defect cells -> {a.paths[0]} | {time.time() - t0:.0f}s", flush=True)


def wrap_rel(d):
    """Displacement rel. difference ours vs LAMMPS on the 8.9 A region (as in compare)."""
    H = d["cell"]; inv = np.linalg.inv(H)
    def wrap(x):
        f = x @ inv; return (f - np.round(f)) @ H
    u = wrap(d["pos1"] - d["pos0"]); v = wrap(d["ref_pos1"] - d["pos0"]); u -= u.mean(0); v -= v.mean(0)
    near = np.linalg.norm(wrap(d["pos0"] - d["site"]), axis=1) <= 8.9
    nrm = np.linalg.norm(v[near])
    return float(np.linalg.norm((u - v)[near]) / nrm) if nrm > 1e-6 else np.nan


def cell_symbols(d):
    """Chemical symbols of a pilot defect cell (types from its unrelaxed .data, in dump order)."""
    A, B = d["folder"].split("_", 1)[1].split("-")
    _, ids, types, _ = read_data(os.path.join(SIM, d["folder"], f"unrelaxed_{d['key']}.data"))
    tm = dict(zip(ids, types))
    return [(A, B)[tm[int(i)] - 1] for i in d["ids"]]


def compare(a):
    B, D = [], []
    for p in sorted(glob.glob(a.paths[0])):
        o = pickle.load(open(p, "rb")); B += o["bulk"]; D += o["defect"]
    da = np.array([np.abs(b["abc"] - b["ref_abc"]).max() for b in B])
    de = np.array([abs(b["e_atom"] - b["ref_e_atom"]) for b in B])
    print(f"bulk: {len(B)} compounds | max |a - a_ref| median {da.mean():.2e} max {da.max():.2e} A | "
          f"|E/atom - ref| median {np.median(de):.2e} max {de.max():.2e} eV | converged {np.mean([b['conv'] for b in B]):.2f}")
    rows = []
    for d in D:
        H = d["cell"]; inv = np.linalg.inv(H)
        def wrap(x):
            f = x @ inv; return (f - np.round(f)) @ H
        u = wrap(d["pos1"] - d["pos0"]); v = wrap(d["ref_pos1"] - d["pos0"])
        u -= u.mean(0); v -= v.mean(0)
        vac = "--1-" in d["key"]
        near = np.linalg.norm(wrap(d["pos0"] - d["site"]), axis=1) <= 8.9  # defect site from basefile.data
        nrm = np.linalg.norm(v[near])
        rows.append(dict(kind="vacancy" if vac else ("pristine" if d["key"].split("-")[1] == d["key"].split("-")[2] else "antisite"),
                         rel=float(np.linalg.norm((u - v)[near]) / nrm) if nrm > 1e-6 else np.nan,
                         maxdiff=float(np.linalg.norm(u - v, axis=1).max()), maxdisp=float(np.linalg.norm(v, axis=1).max()),
                         dE=d["dE"], ref_dE=d["ref_dE"], e0diff=abs(d["e0"] - d["ref_e0"]), conv=d["conv"], sec=d["sec"],
                         natoms=d["natoms"]))
    for kind in ("antisite", "vacancy", "pristine"):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        rel = np.array([x["rel"] for x in r]); ed = np.array([abs(x["dE"] - x["ref_dE"]) for x in r])
        er = np.array([abs(x["dE"] - x["ref_dE"]) / abs(x["ref_dE"]) for x in r if abs(x["ref_dE"]) > 1e-3])
        print(f"{kind:9s} n={len(r):3d} | displacement rel. diff (8.9 A region) median {np.nanmedian(rel):.4f} 90% "
              f"{np.nanpercentile(rel, 90):.4f} max {np.nanmax(rel):.4f} | max atom diff median "
              f"{np.median([x['maxdiff'] for x in r]):.1e} max {max(x['maxdiff'] for x in r):.1e} A (max disp median "
              f"{np.median([x['maxdisp'] for x in r]):.3f} A) | dE diff median {np.median(ed) * 1e3:.2f} meV, rel "
              f"{np.median(er) if len(er) else np.nan:.4f} | E0 diff max {max(x['e0diff'] for x in r):.1e} eV | "
              f"conv {np.mean([x['conv'] for x in r]):.2f} | {np.median([x['sec'] for x in r]):.0f} s/cell")
    bad = sorted(rows, key=lambda x: -np.nan_to_num(x["rel"]))[:5]
    print("worst:", [(round(x["rel"], 3), x["kind"], round(x["maxdisp"], 3)) for x in bad])
    if a.out:
        pickle.dump(dict(bulk=B, rows=rows), open(a.out, "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["relax", "compare"]); ap.add_argument("paths", nargs="+")
    ap.add_argument("--n-compounds", type=int, default=30); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1); ap.add_argument("--fmax", type=float, default=1e-4)
    ap.add_argument("--rattle", type=float, default=0.02, help="sigma (A) of random displacements before relaxing; 0 = pilot setting")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_relax(a) if a.what == "relax" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
