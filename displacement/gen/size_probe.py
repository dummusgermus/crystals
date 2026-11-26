"""GRACE memory / time probe for large C15 cells (size-generalisation test planning).

For C15 supercells n1 x n2 x n3 (24 atoms each) of one compound, with one 8a antisite: peak GPU memory and wall
time of one energy+force call (after a warm-up call) and of 5 FIRE steps. Each size runs in its own subprocess
(an OOM does not kill the probe, peak memory is per size); stops after the first failure.

    TF_FORCE_GPU_ALLOW_GROWTH=true python -I displacement/gen/size_probe.py [A-B]        (GRACE venv)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

SIZES = [(5, 5, 5), (5, 5, 10), (8, 8, 8), (10, 10, 12), (14, 14, 14), (17, 17, 17), (20, 20, 20), (23, 23, 24)]


def one(comp, n):
    import numpy as np
    import tensorflow as tf
    from ase.optimize import FIRE
    from ase.spacegroup import crystal
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from point_pilot import A0_TEMPLATE, POT
    from tensorpotential.calculator import TPCalculator
    A, B = comp.split("-")
    at = crystal([A, B], basis=[(0, 0, 0), (5 / 8, 5 / 8, 5 / 8)], spacegroup=227, cellpar=[A0_TEMPLATE] * 3 + [90] * 3)
    at = at.repeat(n)
    i = int(np.where(np.array(at.get_chemical_symbols()) == A)[0][0]); at[i].symbol = B  # 8a antisite
    at.calc = TPCalculator(POT)
    t = time.time(); at.get_potential_energy(); at.get_forces(); t_first = time.time() - t
    at.positions += np.random.default_rng(0).normal(0, 0.01, at.positions.shape); t = time.time(); at.get_forces()
    t_call = time.time() - t
    t = time.time(); FIRE(at, logfile=None).run(fmax=1e-4, steps=5); t_step = (time.time() - t) / 5
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
    print("RESULT " + json.dumps(dict(n=n, atoms=len(at), peak_gb=peak, t_first=t_first, t_call=t_call, t_step=t_step)), flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        one(sys.argv[2], tuple(int(x) for x in sys.argv[3].split("x"))); return
    comp = sys.argv[1] if len(sys.argv) > 1 else "Ag-Be"
    for n in SIZES:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--one", comp, "x".join(map(str, n))],
                           capture_output=True, text=True)
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if res:
            print(res[0][7:], flush=True)
        else:
            err = [l for l in (p.stderr or "").splitlines() if "Error" in l or "OOM" in l or "memory" in l.lower()]
            print(json.dumps(dict(n=n, atoms=24 * n[0] * n[1] * n[2], failed=True, err=err[-3:])), flush=True)
            break


if __name__ == "__main__":
    main()
