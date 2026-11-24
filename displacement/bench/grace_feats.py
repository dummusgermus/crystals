"""Stage 1 of the external benchmarks (run with the GRACE venv, /work/log1/erik.mueller/grace-venv).

Reads the authors' unrelaxed/relaxed CIF pairs with their splits, reproduces their preprocessing
(DefiNet: vacancies inserted as dummy atoms of the pristine element with defect marker 2,
substitutions marked 1), computes GRACE (GRACE-2L-UEA-OMAT-large, the potential of our own data)
single-point forces, per-atom energies and stress on the REAL unrelaxed atoms (dummies get 0), and the
periodic radius-6 A neighbour list of the unrelaxed structure (incl. dummies).

    python -I displacement/bench/grace_feats.py <dataset> <out.pkl> [--limit N]

dataset: c2db | jarvis | vdW | mp | xmno | definet_low | definet_high
Raw data is expected under displacement/bench_data/ext/<archive name>/.
Output: list of dicts (numpy arrays), one per structure.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from ase.build import make_supercell
from ase.io import read
from ase.neighborlist import neighbor_list

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(os.path.dirname(HERE), "bench_data", "ext")
POT = os.path.join(CODE, "Laves_Screen", "STATIC_FILES", "POTENTIALS", "GRACE-2L-UEA-OMAT-large")
RADIUS = 6.0

DEFINET_HIGH = ["BP_spin_500", "GaSe_spin_500", "hBN_spin_500", "InSe_spin_500", "MoS2_500", "WSe2_500"]
DEFINET_LOW = ["WSe2", "MoS2"]
UNIT = {"WSe2": "WSe2.cif", "MoS2": "MoS2.cif", "InSe": "InSe.cif", "hBN": "BN.cif", "GaSe": "GaSe.cif",
        "BP": "P.cif"}


def structure_to_dict(structure, precision=3):  # authors' helper
    return {tuple(np.round(site.scaled_position, precision)): site for site in structure}


def find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} under {root}")
    return sorted(hits, key=len)[0]


def items_e3relax(ds):
    root = os.path.join(EXT, ds)
    base = os.path.dirname(find(root, "train.csv")) if ds != "xmno" else None
    out = []
    if ds == "xmno":
        for split in ("train", "val", "test"):
            f = find(root, f"id_prop_{split}_all.csv")
            ids = [r.split(",")[0] for r in open(f).read().splitlines() if r.split(",")[0].endswith("relaxed")
                   and not r.split(",")[0].endswith("unrelaxed")]
            croot = os.path.dirname(f)
            cdir = os.path.join(croot, "CIF") if os.path.isdir(os.path.join(croot, "CIF")) else croot
            for i in ids:
                out.append(dict(id=i, split=split, u=os.path.join(cdir, i.replace("relaxed", "unrelaxed") + ".cif"),
                                r=os.path.join(cdir, i + ".cif")))
        return out
    col, fu, fr = ("mp_id", "{}_U.cif", "{}_R.cif") if ds == "mp" else ("atoms_id", "{}_unrelaxed.cif", "{}_relaxed.cif")
    for split in ("train", "val", "test"):
        for i in pd.read_csv(os.path.join(base, f"{split}.csv"))[col]:
            out.append(dict(id=str(i), split=split, u=os.path.join(base, "CIF", fu.format(i)),
                            r=os.path.join(base, "CIF", fr.format(i))))
    return out


def items_definet(which):
    root = os.path.join(EXT, "DefiNet")
    sub = find(root, "high_density_defects" if which == "high" else "low_density_defects") \
        if glob.glob(os.path.join(root, "**", "*_density_defects"), recursive=True) else root
    out = []
    for mat in (DEFINET_HIGH if which == "high" else DEFINET_LOW):
        mroot = os.path.join(sub, mat)
        cdir = os.path.join(mroot, "CIF_POSCAR" if which == "low" else "CIF")
        unit = read(os.path.join(sub, UNIT[mat.split("_")[0]]))
        if which == "high":  # authors: c set to 20 A; 8x8x1 for MoS2/WSe2/hBN, else 6x6x1
            cl = unit.cell.cellpar(); cl[2] = 20; unit.set_cell(cl)
            n = 8 if mat in ("MoS2_500", "WSe2_500", "hBN_spin_500") else 6
        else:
            n = 8
        ideal = structure_to_dict(make_supercell(unit, np.diag([n, n, 1])))
        for split in ("train", "val", "test"):
            for i in pd.read_csv(os.path.join(mroot, f"{split}.csv"))["atoms_id"]:
                out.append(dict(id=f"{mat}/{i}", mat=mat, split=split, ideal=ideal,
                                u=os.path.join(cdir, f"{i}_unrelaxed.cif"), r=os.path.join(cdir, f"{i}_relaxed.cif")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--relax-test", action="store_true",
                    help="also relax every test structure with GRACE (FIRE, fmax 0.02 eV/A, <= 500 steps; the cell "
                         "relaxes too (FrechetCellFilter) when the reference cell changed) -> baseline")
    a = ap.parse_args()
    items = items_definet(a.dataset.split("_")[1]) if a.dataset.startswith("definet") else items_e3relax(a.dataset)
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][: a.limit]
    print(f"{a.dataset}: {len(items)} structures", flush=True)
    from tensorpotential.calculator import TPCalculator
    from tensorpotential import constants as C
    calc = TPCalculator(POT, extra_properties=[C.PREDICT_ATOMIC_ENERGY], truncate_extras_by_natoms=True)
    out, t0, bad = [], time.time(), 0
    for k, it in enumerate(items):
        try:
            au, ar = read(it["u"]), read(it["r"])
        except Exception as e:  # noqa: BLE001
            bad += 1; print("read error", it["id"], e, flush=True); continue
        if len(au) != len(ar) or (au.numbers != ar.numbers).any():
            bad += 1; print("mismatch", it["id"], flush=True); continue
        au.pbc = ar.pbc = True
        # GRACE single point on the real unrelaxed atoms
        au.calc = calc
        f = au.get_forces().astype(np.float32)
        pe = np.asarray(calc.results[C.PREDICT_ATOMIC_ENERGY], dtype=np.float32).ravel()[: len(au)]
        st = au.get_stress(voigt=False).astype(np.float32)  # eV/A^3
        e_tot = float(au.get_potential_energy())
        relaxed = None
        if a.relax_test and it["split"] == "test":
            from ase.filters import FrechetCellFilter
            from ase.optimize import FIRE
            ax = au.copy(); ax.calc = calc
            cell_moves = not np.allclose(np.array(au.cell), np.array(ar.cell), atol=1e-4)
            opt = FIRE(FrechetCellFilter(ax) if cell_moves else ax, logfile=None)
            try:
                conv = bool(opt.run(fmax=0.02, steps=500))
            except Exception as e:  # noqa: BLE001
                print("relax error", it["id"], e, flush=True); conv = False
            relaxed = dict(pos=ax.positions.astype(np.float64), cell=np.array(ax.cell, np.float64), conv=conv,
                           steps=int(opt.nsteps))
            ax.calc = None
        au.calc = None
        n_real = len(au)
        defect = None
        if "ideal" in it:  # DefiNet scheme
            dd = structure_to_dict(au)
            defect = [0] * n_real
            for key, site in it["ideal"].items():
                if key not in dd:
                    au.append(site); ar.append(site); defect.append(2)
                elif site.number != dd[key].number:
                    defect[dd[key].index] = 1
            defect = np.array(defect, dtype=np.int8)
        n = len(au)
        F = np.zeros((n, 3), np.float32); F[:n_real] = f
        PE = np.zeros(n, np.float32); PE[:n_real] = pe
        i, j, S, d = neighbor_list("ijSd", au, RADIUS)
        out.append(dict(id=it["id"], mat=it.get("mat", ""), split=it["split"], z=au.numbers.astype(np.int16),
                        pos_u=au.positions.astype(np.float64), pos_r=ar.positions.astype(np.float64),
                        cell_u=np.array(au.cell, np.float64), cell_r=np.array(ar.cell, np.float64),
                        defect=defect, n_real=n_real, force0=F, pe0=PE, stress0=st, e0=e_tot,
                        grace_relaxed=relaxed, ei=i.astype(np.int32), ej=j.astype(np.int32), eS=S.astype(np.int8), ed=d.astype(np.float32)))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(items)} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(f"done {len(out)} ok, {bad} skipped | {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
