"""Compare the ASE/GRACE relaxation (grace_relax.py validate) with the LAMMPS labels of the planar data.
    python displacement/mixed/check_protocol.py <validate.pkl>
Per cell: single-point agreement (pe0, F0), non-affine displacement error relative to |d_LAMMPS| (as r_disp),
c-vector change, relaxation energy."""
import pickle, sys
import numpy as np


def nonaffine(p0, H0, p1, H1):
    s = p1 @ np.linalg.inv(H1) - p0 @ np.linalg.inv(H0); s -= np.round(s)
    d = s @ H0; return d - d.mean(0)


rows = []
for r in pickle.load(open(sys.argv[1], "rb")):
    dl = nonaffine(r["ref_pos0"], r["ref_cell0"], r["ref_pos1"], r["ref_cell1"])
    da = nonaffine(r["ref_pos0"], r["ref_cell0"], r["pos1"], r["cell1"])
    rd = np.linalg.norm(da - dl) / max(np.linalg.norm(dl), 1e-9)
    dc_l = r["ref_cell1"][2] - r["ref_cell0"][2]; dc_a = r["cell1"][2] - r["ref_cell0"][2]
    eL = float(r["ref_pe1"].sum() - r["ref_pe0"].sum()); eA = float(r["pe1"].sum() - r["pe0"].sum())
    rows.append((np.abs(r["pe0"] - r["ref_pe0"]).max(), np.abs(r["force0"] - r["ref_force0"]).max(), rd,
                 np.linalg.norm(dl, axis=1).max(), np.linalg.norm(dc_a - dc_l), np.linalg.norm(dc_l), eA - eL, eL))
    print(f"{r['id']:24s} pe0 {rows[-1][0]:.1e} F0 {rows[-1][1]:.1e} | disp rel.err {rd:.3f} (max|d| {rows[-1][3]:.3f} A) | "
          f"dc err {rows[-1][4]:.4f} of {rows[-1][5]:.4f} A | dE {eA:.4f} vs {eL:.4f} eV | conv {r['conv']}")
a = np.array(rows)
print(f"\nmedian: single point pe0 {np.median(a[:, 0]):.1e} eV, F0 {np.median(a[:, 1]):.1e} eV/A | displacement rel. err "
      f"{np.median(a[:, 2]):.3f} | dc err {np.median(a[:, 4]):.4f} A | |dE err| {np.median(np.abs(a[:, 6])):.4f} eV "
      f"(rel {np.median(np.abs(a[:, 6]) / np.maximum(np.abs(a[:, 7]), 1e-3)):.3f})")
