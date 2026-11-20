"""Minimal LAMMPS dump reader (single frame, triclinic) without ovito."""
import numpy as np


def read_dump(path):
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[3]); hdr = lines[4].split()
    b = np.array([[float(v) for v in lines[5 + k].split()] for k in range(3)])
    if len(hdr) > 6:  # triclinic: xlo_bound xhi_bound xy / ylo_b yhi_b xz / zlo_b zhi_b yz
        xy, xz, yz = b[0, 2], b[1, 2], b[2, 2]
    else:
        xy = xz = yz = 0.0
    xlo = b[0, 0] - min(0.0, xy, xz, xy + xz); xhi = b[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = b[1, 0] - min(0.0, yz); yhi = b[1, 1] - max(0.0, yz)
    zlo, zhi = b[2, 0], b[2, 1]
    cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])  # rows = a, b, c
    origin = np.array([xlo, ylo, zlo])
    cols = lines[8].split()[2:]
    arr = np.array([l.split() for l in lines[9:9 + n]], dtype=float)
    d = {c: arr[:, k] for k, c in enumerate(cols)}
    ids = d["id"].astype(int); o = np.argsort(ids)
    out = {"ids": ids[o], "cell": cell, "origin": origin}
    pk = ("xu", "yu", "zu") if "xu" in d else ("x", "y", "z")
    pos = np.stack([d[k] for k in pk], 1)
    if "ix" in d and pk[0] == "x":
        img = np.stack([d["ix"], d["iy"], d["iz"]], 1)
        pos = pos + img @ cell
    out["pos"] = pos[o]
    out["force"] = np.stack([d["fx"], d["fy"], d["fz"]], 1)[o]
    pe = d.get("c_pe_potential_energy", d.get("c_pe"))
    out["pe"] = pe[o]
    if "type" in d:
        out["type"] = d["type"][o].astype(int)
    return out
