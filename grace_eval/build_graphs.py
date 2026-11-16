"""Build PyG graphs for the GRACE C15 evaluation set (laves-ase-grace-c15) in exactly the
planar training format (planar_graph_maker._build_full_graph + cycle features + labels).

Sets (per compound):
  relax  : <S>-p0-initial -> <S>-p0 for S in FFHFF, Primitive, XYZ
           (initial geometry + initial per-atom PE as input, y = PE_relaxed - PE_initial).
  strain : strained single-point configurations (n10, n5, p5, p10) and p0 as the input
           structure; target = relaxed p0 per-atom PE (same atom ids). This encodes the
           colleague's expectation that relaxing any strained cell returns to p0.
Outputs grace_eval/graphs_{relax,strain}.pt and grace_eval/build_log.json."""
import glob, json, os, re, sys, time
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_graph_maker as pg
from graph_maker import EDGE_K, SYMBOL_TO_Z
from adv_graph_maker import count_cycles_per_node
from atom_encodings import element_table
from stacking_labels import defect_mask

DATA = os.path.join(ROOT, "laves-ase-grace-c15", "laves-ase-grace-c15")
OUT = os.path.dirname(os.path.abspath(__file__))
# training normalisation of the 3/4-cycle counts (recovered exactly from the planar training set)
CYC_MEAN = np.array([6.930240, 24.800112]); CYC_STD = np.array([4.402056, 22.009208])
STRAINS = ["n10", "n5", "p0", "p5", "p10"]
STRAIN_PCT = {"n10": -10, "n5": -5, "p0": 0, "p5": 5, "p10": 10}


def read_atoms(path):
    from ovito.io import import_file
    d = import_file(path).compute()
    return np.asarray(d.particles.positions), np.asarray(d.particles["Particle Type"]), np.asarray(d.cell.matrix)[:, :3]


def read_dump(path):
    """(header lines, atom rows [id, type, x, y, z, pe]) of a LAMMPS text dump."""
    lines = open(path).read().splitlines()
    i = lines.index(next(l for l in lines if l.startswith("ITEM: ATOMS")))
    return lines[: i + 1], np.array([[float(v) for v in l.split()] for l in lines[i + 1:] if l.strip()])


def primitive_as_hex(prim_path, xyz_path, out_path):
    """Pristine C15 (no internal degrees of freedom, per-type PE uniform to <1e-6 eV) re-expressed
    in the 18-atom hexagonal cell used by the training graphs: XYZ-p0 geometry scaled isotropically
    to the Primitive configuration's volume per atom, per-type PE taken from the Primitive dump."""
    _, P = read_dump(prim_path); hdr, X = read_dump(xyz_path)
    _, _, cp = read_atoms(prim_path); _, _, cx = read_atoms(xyz_path)
    s = ((abs(np.linalg.det(cp)) / len(P)) / (abs(np.linalg.det(cx)) / len(X))) ** (1 / 3)
    pe = {t: P[P[:, 1] == t, 5].mean() for t in (1, 2)}
    assert max(P[P[:, 1] == t, 5].std() for t in (1, 2)) < 1e-4, "Primitive PE not uniform per type"
    out = X.copy(); out[:, 2:5] *= s; out[:, 5] = [pe[int(t)] for t in X[:, 1]]
    box = []
    for l in hdr[5:8]:
        box.append(" ".join(f"{float(v) * float(s):.17g}" for v in l.split()))
    with open(out_path, "w") as fh:
        fh.write("\n".join(hdr[:5] + box + hdr[8:]) + "\n")
        for r in out:
            fh.write(f"{int(r[0])} {int(r[1])} {r[2]:.17g} {r[3]:.17g} {r[4]:.17g} {r[5]:.17g}\n")
    return out_path


_LABELS = {}


def stacking_labels(path):
    if path not in _LABELS:
        from ovito.io import import_file
        d = import_file(path).compute()
        pos, types = np.asarray(d.particles.positions), np.asarray(d.particles["Particle Type"])
        pid = np.asarray(d.particles["Particle Identifier"])
        m, steps = defect_mask(pos, types, np.asarray(d.cell.matrix)[:, :3].T)
        _LABELS[path] = ([int(p) for p in pid[m]], steps)
    return _LABELS[path]


def make_graph(comp, a, b, struct, cond, init_path, relaxed_path):
    pos, types, cell = read_atoms(init_path)
    if struct == "Primitive":  # pristine C15 in the hexagonal training cell: no stacking deviation
        tmp = os.path.join(OUT, "tmp_hex"); os.makedirs(tmp, exist_ok=True)
        xyz = init_path.split("-C15-")[0] + "-C15-XYZ-p0.dump"
        init_path = primitive_as_hex(init_path, xyz, os.path.join(tmp, f"{comp}_{cond}_init.dump"))
        relaxed_path = primitive_as_hex(relaxed_path, xyz, os.path.join(tmp, f"{comp}_{cond}_relaxed.dump"))
        pos, types, cell = read_atoms(init_path)
        ids, steps = [], "pristine"
    else:  # labels from the unrelaxed p0 geometry of this structure, reused by atom id for every condition
        ids, steps = stacking_labels(init_path.split("-C15-")[0] + f"-C15-{struct}-p0-initial.dump")
        pos, types, cell = read_atoms(init_path)
    x, p, ei, ea, y, meta, pids = pg._build_full_graph(
        init_path, relaxed_path, init_path, ids or None, EDGE_K, 3.0, "shell", target_mode="residual")
    cyc = count_cycles_per_node(ei, x.size(0), max_cycle_len=4)[:, :2]
    x = torch.cat([x, torch.tensor((cyc - CYC_MEAN) / CYC_STD, dtype=torch.float)], dim=-1)
    tz = {1: SYMBOL_TO_Z[a], 2: SYMBOL_TO_Z[b]}
    g = Data(x=x, edge_index=ei, edge_attr=ea, pos=p, y=y,
             z=torch.tensor([tz[int(t)] for t in x[:, 0].long().tolist()], dtype=torch.long))
    g.folder = f"{a}-{b}_{struct}_{cond}"; g.element_a, g.element_b = a, b
    g.stack_sequence = struct; g.condition = cond; g.strain_pct = STRAIN_PCT.get(cond.replace("-initial", ""), 0)
    g.edge_k, g.edge_radius, g.edge_mode, g.target_mode = EDGE_K, 3.0, "shell", "residual"
    g.meta = {**meta, "stack_steps": steps}; g.particle_ids = pids
    g.delta_total_eV = torch.tensor([meta["delta_total_eV"]], dtype=torch.float)
    g.graph_delta_total_eV = torch.tensor([meta["graph_delta_total_eV"]], dtype=torch.float)
    g.cell_volume = float(abs(np.linalg.det(cell)))
    return g


def main():
    known = set(element_table().z_vocab)
    relax, strain, log = [], [], {"skipped": [], "errors": []}
    comps = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    t0 = time.time()
    for n, comp in enumerate(comps):
        m = re.fullmatch(r"([A-Z][a-z]?)([A-Z][a-z]?)2", comp)
        a, b = m.groups()
        if SYMBOL_TO_Z.get(a) not in known or SYMBOL_TO_Z.get(b) not in known:
            log["skipped"].append({"compound": comp, "reason": "element missing from element_properties.json"}); continue
        pre = os.path.join(DATA, comp, f"{a}{b}-C15-")
        for struct in ("FFHFF", "Primitive", "XYZ"):
            try:
                relax.append(make_graph(comp, a, b, struct, "p0", pre + f"{struct}-p0-initial.dump", pre + f"{struct}-p0.dump"))
                if struct != "XYZ":
                    for c in STRAINS:
                        strain.append(make_graph(comp, a, b, struct, c, pre + f"{struct}-{c}.dump", pre + f"{struct}-p0.dump"))
            except Exception as e:  # noqa: BLE001
                log["errors"].append({"compound": comp, "struct": struct, "error": repr(e)[:300]})
        if (n + 1) % 20 == 0:
            print(f"[{n + 1}/{len(comps)}] {len(relax)} relax / {len(strain)} strain graphs, "
                  f"{len(log['errors'])} errors, {time.time() - t0:.0f}s", flush=True)
    torch.save(relax, os.path.join(OUT, "graphs_relax.pt")); torch.save(strain, os.path.join(OUT, "graphs_strain.pt"))
    log.update(n_relax=len(relax), n_strain=len(strain), n_compounds=len(comps))
    json.dump(log, open(os.path.join(OUT, "build_log.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in log.items()}), flush=True)


if __name__ == "__main__":
    main()
