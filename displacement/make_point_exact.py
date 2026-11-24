"""Point-defect energy labels from the exact LAMMPS totals.
    python displacement/make_point_exact.py   -> data/superset_point_exact.pt (data/superset_point.pt untouched)
The per-atom pe in the .dump files has ~5 significant digits, so sum(pe1 - pe0) is noisy (median 3 meV, max 88 meV
over 3000 atoms). The exact cell totals are in <unrelaxed|relaxed>_<key>.dump.yaml (pe:). Per graph:
  dE_exact = pe(relaxed yaml) - pe(unrelaxed yaml)
  pe1 += (dE_exact - sum_graph(pe1 - pe0)) / n_core   on the n_core atoms within 8.9 A of the defect (inside every
                                                       training view, rt >= 8.9); pe0 = model input, unchanged
so the graph-total target is the exact full-cell relaxation energy (rounding noise removed; the far-field tail
outside the 14 A graph, p90 ~25 meV, is folded into the core region). Per-atom targets move by ~0.05 meV.
Stable cells whose far-field share is large, |dE_exact - sum_graph| > max(50 meV, 10% |dE|) (8 cells, borderline
global relaxations, far-field median displacement just under the 0.05 A unstable cut), are flagged unstable (as
global rearrangements already are), unstable_orig keeps the old flag. Original pe1 in pe1_dump, dump total in dE_total.
"""
import os
import re

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PE = re.compile(r"^\s*pe:\s*(\S+)", re.M)


def yaml_pe(path):
    m = PE.findall(open(path).read())
    return float(m[-1])  # last step = final state


def main():
    G = torch.load(os.path.join(HERE, "data", "superset_point.pt"), weights_only=False)
    corr, noise, flagged = [], [], []
    for g in G:
        d = os.path.join(ROOT, "SIMULATIONS", g.folder)
        e = yaml_pe(f"{d}/relaxed_{g.defect_key}.dump.yaml") - yaml_pe(f"{d}/unrelaxed_{g.defect_key}.dump.yaml")
        view = float((g.pe1.double() - g.pe0.double()).sum())
        core = g.dist <= 8.9
        g.pe1_dump = g.pe1.clone()
        pe1 = g.pe1.double(); pe1[core] += (e - view) / int(core.sum()); g.pe1 = pe1.to(g.pe1_dump.dtype)
        g.dE_exact = e
        g.unstable_orig = bool(g.unstable)
        if not g.unstable and abs(e - view) > max(0.05, 0.1 * abs(e)):
            g.unstable = True; flagged.append(f"{g.folder} {g.defect_key} {e - view:+.3f} eV")
        if not g.unstable_orig:
            corr.append(e - view); noise.append(e - g.dE_total)
    torch.save(G, os.path.join(HERE, "data", "superset_point_exact.pt"))
    c, n = 1e3 * np.abs(corr), 1e3 * np.abs(noise)
    print("newly flagged unstable:", len(flagged), *flagged, sep="\n  ")
    print(f"{len(G)} graphs, {len(corr)} stable before flagging | |exact - view sum| median {np.median(c):.2f} meV, p90 {np.percentile(c, 90):.1f}, max {c.max():.1f} "
          f"| |exact - full dump sum| median {np.median(n):.2f} meV, max {n.max():.1f}")


if __name__ == "__main__":
    main()
