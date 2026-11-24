"""Write task files for the displacement study (4 trainings per GPU task line).

    python displacement/make_tasks.py phase1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cmd(domain, model, view, head="eqv", loss="mse", seed=0, epochs=400, extra="", group="disp-phase1", out="runs"):
    return (f"python -u displacement/train_disp.py --domain {domain} --model {model} --view \"{view}\" "
            f"--head {head} --loss {loss} --seed {seed} --epochs {epochs} --group {group} "
            f"--out-dir displacement/{out} {extra}").strip()


def phase1():
    """One-factor-at-a-time around a production-like centre, plus the same around a
    'force' centre (initial forces as input), for both models and 2 seeds."""
    confs = []  # (domain, view, head, loss, extra)
    for dom in ("point", "planar"):
        reg = "rt=8.9,buf=4," if dom == "point" else ""
        C = dict(edge="k3", feat="base+cyc")
        def v(**kw):
            d = dict(C); d.update(kw)
            return reg + ",".join(f"{k}={x}" for k, x in d.items())
        # centre
        confs.append((dom, v(), "eqv", "mse", ""))
        # node / edge features
        for f in ("base", "base+cyc+force", "base+cyc+dvec", "base+cyc+force+dvec"):
            confs.append((dom, v(feat=f), "eqv", "mse", ""))
        # edges (message passing + head)
        for e in ("k1", "k2", "k5", "k8"):
            confs.append((dom, v(edge=e), "eqv", "mse", ""))
            confs.append((dom, v(edge=e, feat="base+cyc+force"), "eqv", "mse", ""))
        # heads
        confs.append((dom, v(feat="base+cyc+force"), "eqvF", "mse", "--head-force"))
        confs.append((dom, v(feat="base+cyc+force"), "eqv", "mse", "--head-force"))
        confs.append((dom, v(feat="base+cyc+dirs"), "direct", "mse", ""))
        # loss / unstable structures
        for l in ("huber", "l1"):
            confs.append((dom, v(), "eqv", l, ""))
        confs.append((dom, v() + ",stable=1", "eqv", "mse", ""))
        confs.append((dom, v(feat="base+cyc+force") + ",stable=1", "eqv", "mse", ""))
        if dom == "point":
            for rt, buf in ((6.4, 4), (8.9, 0), (10, 4), (11, 3)):
                for f in ("base+cyc", "base+cyc+force"):
                    confs.append((dom, f"rt={rt},buf={buf},edge=k3,feat={f}", "eqv", "mse", ""))
    lines = []
    for model in ("cgcnn", "transformer"):
        for seed in (0, 1):
            for dom, view, head, loss, extra in confs:
                lines.append(cmd(dom, model, view, head, loss, seed, extra=extra))
    return lines


def phase1b():
    """Full periodic cell vs cut-out cluster (point) and the PaiNN equivariant backbone."""
    heavy, light = [], []
    g = "disp-phase1b"; sp = "--save-preds"
    for seed in (0, 1):
        F = "rt=8.9,edge=k3,feat=base+force"
        for model in ("cgcnn", "painn"):
            heavy.append(cmd("point_full", model, "rt=8.9,edge=k3,feat=base", seed=seed, group=g, extra=sp))
            heavy.append(cmd("point_full", model, F, seed=seed, group=g, extra=sp))
            heavy.append(cmd("point_full", model, "rt=0,edge=k3,feat=base+force", seed=seed, group=g, extra=sp))
        heavy.append(cmd("point_full", "cgcnn", F, head="eqvF", seed=seed, group=g, extra=sp + " --head-force"))
        heavy.append(cmd("point_full", "painn", F, head="eqvF", seed=seed, group=g, extra=sp))
        heavy.append(cmd("point_full", "cgcnn", "rt=8.9,edge=k2,feat=base+force", seed=seed, group=g, extra=sp))
        heavy.append(cmd("point_full", "cgcnn", "rt=8.9,edge=k4,feat=base+force", seed=seed, group=g, extra=sp))
        heavy.append(cmd("point_full", "transformer", F, seed=seed, group=g, extra=sp + " --bs 4"))
        for f in ("base+cyc", "base+cyc+force"):
            light.append(cmd("point", "painn", f"rt=8.9,buf=4,edge=k3,feat={f}", seed=seed, group=g, extra=sp))
            light.append(cmd("planar", "painn", f"edge=k3,feat={f}", seed=seed, group=g, extra=sp))
        light.append(cmd("point", "painn", "rt=8.9,buf=4,edge=k3,feat=base+cyc+force", head="eqvF", seed=seed, group=g, extra=sp))
        light.append(cmd("planar", "painn", "edge=k3,feat=base+cyc+force", head="eqvF", seed=seed, group=g, extra=sp))
        light.append(cmd("planar", "painn", "edge=k2,feat=base+cyc+force", head="eqvF", seed=seed, group=g, extra=sp))
    return heavy, light


def phase2():
    """Sharpen around the phase-1 winners: CGCNN + force projections in the head.
    3 compound splits x 2 seeds per config; everything scored on the same region."""
    g = "disp-phase2"; out = "runs_p2"
    P = "rt=8.9,buf=4,"
    confs = []  # (domain, model, view, head, extra)
    HF = "--head-force"
    # ---- point (cluster), CGCNN
    for f in ("base+cyc+force", "base+force"):
        confs.append(("point", "cgcnn", P + f"edge=k3,feat={f}", "eqv", HF))
    confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc", "eqv", ""))               # phase-1 centre
    confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc+force", "eqvF", HF))
    confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc+force,stable=1", "eqv", HF))
    confs.append(("point", "cgcnn", P + "edge=k2,feat=base+cyc+force", "eqv", HF))
    confs.append(("point", "cgcnn", "rt=6.4,buf=4,edge=k3,feat=base+cyc+force", "eqv", HF))
    confs.append(("point", "cgcnn", "rt=6.4,buf=4,edge=k3,feat=base+cyc+force,stable=1", "eqv", HF))
    confs.append(("point", "cgcnn", "rt=8.9,buf=6,edge=k3,feat=base+cyc+force", "eqv", HF))
    confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc+force", "eqv", HF + " --hidden 256 --layers 6 --tag h256l6"))
    confs.append(("point", "transformer", P + "edge=k2,feat=base+cyc+force", "eqv", HF))
    confs.append(("point", "transformer", P + "edge=k3,feat=base+cyc+force", "eqv", HF))
    # ---- planar, CGCNN
    for f in ("base+cyc+force", "base+force"):
        confs.append(("planar", "cgcnn", f"edge=k3,feat={f}", "eqv", HF))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc", "eqv", ""))                  # phase-1 centre
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force", "eqvF", HF))
    confs.append(("planar", "cgcnn", "edge=k2,feat=base+cyc+force", "eqv", HF))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force,stable=1", "eqv", HF))
    confs.append(("planar", "cgcnn", "edge=k3,hedge=k5,feat=base+cyc+force", "eqv", HF))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force", "eqv", HF + " --layers 6 --tag l6"))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force", "eqv", HF + " --hidden 256 --layers 6 --tag h256l6"))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force", "eqv", HF + " --wd 1e-3 --tag wd1e-3"))
    confs.append(("planar", "transformer", "edge=k3,feat=base+cyc+force", "eqv", HF))
    lines = []
    for sp in (0, 1, 2):
        for seed in (0, 1):
            for dom, model, view, head, extra in confs:
                lines.append(cmd(dom, model, view, head, "mse", seed, group=g, out=out,
                                 extra=f"--split-seed {sp} --save-preds {extra}"))
    return lines


def phase2b():
    """Stabilisers against the out-of-distribution blow-ups seen in phase 2 (predictions of
    10-300 A on some test compounds): LayerNorm in the CGCNN stack, standardised inputs."""
    g = "disp-phase2b"; out = "runs_p2"; HF = "--head-force"
    P = "rt=8.9,buf=4,"
    NS = {"norm": "--norm", "std": "--std-inputs", "both": "--norm --std-inputs"}
    confs = []
    for k in ("norm", "std", "both"):
        confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc+force,stable=1", "eqv", HF + " " + NS[k]))
        confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force", "eqv", HF + " " + NS[k]))
    confs.append(("point", "cgcnn", P + "edge=k3,feat=base+cyc+force", "eqv", HF + " " + NS["both"]))
    confs.append(("point", "cgcnn", P + "edge=k2,feat=base+cyc+force", "eqv", HF + " " + NS["both"]))
    confs.append(("point", "transformer", P + "edge=k2,feat=base+cyc+force", "eqv", HF + " " + NS["std"]))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+cyc+force,stable=1", "eqv", HF + " " + NS["both"]))
    confs.append(("planar", "cgcnn", "edge=k3,feat=base+force", "eqv", HF + " " + NS["both"]))
    confs.append(("planar", "transformer", "edge=k3,feat=base+cyc+force", "eqv", HF + " " + NS["std"]))
    lines = []
    for sp in (0, 1, 2):
        for seed in (0, 1):
            for dom, model, view, head, extra in confs:
                lines.append(cmd(dom, model, view, head, "mse", seed, group=g, out=out,
                                 extra=f"--split-seed {sp} --save-preds {extra}"))
    return lines


def _p3(confs, g="disp-phase3", out="runs_p3"):
    lines = []
    for sp in (0, 1, 2):
        for seed in (0, 1):
            for dom, model, view, head, extra in confs:
                lines.append(cmd(dom, model, view, head, "mse", seed, group=g, out=out,
                                 extra=f"--split-seed {sp} --save-preds {extra}".strip()))
    return lines


def phase3h():
    """PaiNN on full periodic point cells (heavy: ~18 GB, 2 per 48 GB GPU), 3 splits x 2 seeds."""
    return _p3([("point_full", "painn", "rt=8.9,edge=k3,feat=base+force", "eqv", ""),
                ("point_full", "painn", "rt=8.9,edge=k3,feat=base+force,stable=1", "eqv", ""),
                ("point_full", "painn", "rt=8.9,edge=k2,feat=base+force", "eqv", "")])


def phase3l():
    """PaiNN on point clusters and planar cells, 3 splits x 2 seeds."""
    return _p3([("point", "painn", "rt=8.9,buf=4,edge=k3,feat=base+force", "eqv", ""),
                ("point", "painn", "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1", "eqv", ""),
                ("planar", "painn", "edge=k3,feat=base+cyc+force", "eqvF", ""),
                ("planar", "painn", "edge=k3,feat=base+cyc+force", "eqv", ""),
                ("planar", "painn", "edge=k3,feat=base+force", "eqvF", ""),
                ("planar", "painn", "edge=k3,feat=base+cyc+force,stable=1", "eqvF", ""),
                ("planar", "painn", "edge=k3,feat=base+cyc+force", "eqvF", "--layers 6 --tag l6"),
                ("planar", "painn", "edge=k3,feat=base+cyc+force", "eqvF", "--wd 1e-3 --tag wd1e-3")])


NS = "--norm --std-inputs"


def phase4h():
    """Stabilised full-cell point models (heavy, 48 GB GPUs, 2 per GPU)."""
    return _p3([("point_full", "painn", "rt=8.9,edge=k3,feat=base+force,stable=1", "eqv", NS),
                ("point_full", "painn", "rt=8.9,edge=k3,feat=base+force,stable=1,fsat=2", "eqv", NS),
                ("point_full", "cgcnn", "rt=8.9,edge=k3,feat=base+force,stable=1", "eqv", "--head-force " + NS)],
               g="disp-phase4", out="runs_p4")


def phase4l():
    """Stabilised PaiNN (scalar LN + vector RMS norm, standardised inputs, optional force cap)."""
    P = "rt=8.9,buf=4,"
    return _p3([("point", "painn", P + "edge=k3,feat=base+cyc+force,stable=1", "eqv", NS),
                ("point", "painn", P + "edge=k3,feat=base+cyc+force,stable=1,fsat=2", "eqv", NS),
                ("point", "cgcnn", P + "edge=k3,feat=base+cyc+force,stable=1,fsat=2", "eqv", "--head-force " + NS),
                ("planar", "painn", "edge=k3,feat=base+cyc+force,stable=1", "eqvF", NS),
                ("planar", "painn", "edge=k3,feat=base+cyc+force,stable=1,fsat=2", "eqvF", NS),
                ("planar", "painn", "edge=k3,feat=base+force,stable=1", "eqvF", NS)],
               g="disp-phase4", out="runs_p4")


def phase5():
    """PaiNN capacity / optimisation around the stabilised best configs."""
    PT = ("point", "painn", "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1,fsat=2", "eqv")
    PL = ("planar", "painn", "edge=k3,feat=base+force,stable=1", "eqvF")
    var = [("--hidden 256 --tag h256"), ("--lr 5e-4 --tag lr5e-4"), ("--layers 3 --tag l3"), ("--layers 5 --tag l5")]
    return _p3([(*c, NS + " " + v) for c in (PT, PL) for v in var], g="disp-phase5", out="runs_p5")


EXT = "--loss l1 --hidden 512 --lr 1e-4 --ema 0.999 --amsgrad"  # authors' recipe (size, lr, EMA, amsgrad, |dr| loss)


def _ext(dom):
    v = "edge=r6n50,feat=base+cyc+force,stable=1" if dom == "planar" else \
        "rt=8.9,buf=4,edge=r6n50,feat=base+cyc+force,stable=1"
    extra = "" if dom == "planar" else " --bs 4 --epochs 250"
    confs = []
    for m in ("definet", "e3relax", "e3relaxh2"):
        confs.append((dom, m, v, "eqv", EXT + extra))                                          # as published
        confs.append((dom, m, v, "eqv", EXT + extra + " --ext-feats --ext-force --std-inputs"))  # + our inputs
    return _p3(confs, g="disp-ext", out="runs_ext")


def ext_planar():
    """DefiNet / E3Relax / E3Relax-H2 on planar faults (3 per 48 GB GPU)."""
    return _ext("planar")


def ext_point():
    """DefiNet / E3Relax / E3Relax-H2 on point-defect clusters (1 per 48 GB GPU)."""
    return _ext("point")


def write(name, lines, per_line=4):
    path = os.path.join(HERE, f"tasks_{name}.txt")
    grouped = [" ;; ".join(lines[i:i + per_line]) for i in range(0, len(lines), per_line)]
    with open(path, "w") as fh:
        fh.write("\n".join(grouped) + "\n")
    print(f"{path}: {len(lines)} runs in {len(grouped)} tasks")


def _abl(dom):
    """Which DefiNet ingredient helps? Base = DefiNet + our inputs (r6n50, h512, l1, lr 1e-4, EMA, amsgrad)."""
    pt = dom != "planar"
    v = "rt=8.9,buf=4,edge={e},feat=base+cyc+force,stable=1" if pt else "edge={e},feat=base+cyc+force,stable=1"
    ours = " --ext-feats --ext-force --std-inputs"
    extra = " --bs 4 --epochs 250" if pt else ""
    D = EXT + extra + ours
    c = [(dom, "definet", v.format(e="r6n50"), "eqv", D + " --ext-static --tag static"),
         (dom, "definet", v.format(e="r6n50"), "eqv", D + " --ext-novn --tag novn"),
         (dom, "definet", v.format(e="k3"), "eqv", D),
         (dom, "definet", v.format(e="r6n50"), "eqv", D.replace("--loss l1", "--loss mse"))]
    if pt:
        c += [(dom, "definet", v.format(e="r6n50"), "eqv", D.replace("--hidden 512", "--hidden 128")),
              (dom, "definet", v.format(e="r6n50"), "eqv", D.replace(" --ema 0.999", "")),
              (dom, "painn", v.format(e="r6n50"), "eqv", NS),
              (dom, "painn", v.format(e="k3"), "eqv", NS + " " + EXT + extra + " --tag dfrecipe")]
    return _p3(c, g="disp-ext2", out="runs_ext2")


def abl_point():
    return _abl("point")


def abl_planar():
    return _abl("planar")


# ---- combined point + planar model (one model for both defect kinds) ----
def joint():
    """Joint point+planar models vs separate models, all on ONE compound split over the union of compounds
    (--union-split; joint always), same view (k3, base+cyc+force, stable). 3 splits x 2 seeds."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    PT = " --bs 4 --epochs 250"
    R = NS + " " + EXT + " --tag dfrecipe"        # PaiNN with the DefiNet recipe (best in the ablation)
    D = EXT + " --ext-feats --ext-force --std-inputs"  # DefiNet + our inputs
    us = " --union-split"
    c = [("point", "painn", V, "eqv", R + PT + us),        # separate baselines
         ("planar", "painn", V, "eqvF", R + us),
         ("point", "definet", V, "eqv", D + PT + us),
         ("planar", "definet", V, "eqv", D + us),
         ("point", "painn", V, "eqv", NS + PT + us),       # previous production recipes
         ("planar", "painn", V, "eqvF", NS + " --lr 5e-4 --tag lr5e-4" + us),
         ("joint", "painn", V, "eqvF", R + PT),            # joint models
         ("joint", "painn", V, "eqv", R + PT),
         ("joint", "definet", V, "eqv", D + PT),
         ("joint", "painn", V, "eqvF", R + PT + " --no-dom-balance"),
         ("joint", "painn", V, "eqvF", R + PT + " --no-domflag")]
    return _p3(c, g="disp-joint", out="runs_joint")


def final():
    """Production position model: joint PaiNN (DefiNet recipe) + joint DefiNet (our inputs), 2 seeds each,
    trained on 90 % of ALL compounds (10 % validation for checkpoint selection, no test set)."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    PT = " --bs 4 --epochs 250 --final --save-preds --split-seed 0"
    out = []
    for seed in (0, 1):
        out.append(cmd("joint", "painn", V, "eqvF", "mse", seed, group="disp-final", out="runs_final",
                       extra=NS + " " + EXT + " --tag dfrecipe" + PT))
        out.append(cmd("joint", "definet", V, "eqv", "mse", seed, group="disp-final", out="runs_final",
                       extra=EXT + " --ext-feats --ext-force --std-inputs" + PT))
    return out


def final_mt():
    """Multi-task production candidate: joint PaiNN (DefiNet recipe) with energy as second output (lambda 0.3,
    best in runs_energy), 2 seeds, same 90/10 all-compound split as final(); out runs_final_mt."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    PT = " --bs 4 --epochs 250 --final --save-preds --split-seed 0 --energy-weight 0.3"
    return [cmd("joint", "painn", V, "eqvF", "mse", seed, group="disp-final", out="runs_final_mt",
                extra=NS + " " + EXT + " --tag dfrecipe" + PT) for seed in (0, 1)]


def unified():
    """One model for point defects, planar faults and cells with both (train_disp --unified: no domain input,
    separate point-defect / fault channels, one output unit). All runs score the mixed cells of their test
    compounds (--mixed eval = zero-shot); --mixed train also trains on the mixed cells of training compounds."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    R = NS + " " + EXT + " --tag dfrecipe --bs 4 --epochs 250 --unified"
    E = " --energy-weight 0.3"
    D = EXT + " --ext-feats --ext-force --std-inputs --bs 4 --epochs 250 --unified"
    c = [("joint", "painn", V, "eqvF", R + " --mixed eval"),
         ("joint", "painn", V, "eqvF", R + " --mixed eval" + E),
         ("joint", "painn", V, "eqvF", R + " --mixed eval" + E + " --e-harm"),
         ("joint", "painn", V, "eqvF", R + " --mixed eval" + E + " --e-harm-detach"),
         ("joint", "painn", V, "eqvF", R + " --mixed train" + E),
         ("joint", "painn", V, "eqvF", R + " --mixed train" + E + " --e-harm"),
         ("joint", "definet", V, "eqv", D + " --mixed eval"),
         ("joint", "definet", V, "eqv", D + " --mixed train")]
    return _p3(c, g="disp-unified", out="runs_unified")


def final_unified():
    """Unified production candidates (best configs of runs_unified), 2 seeds each, trained on the 90/10
    all-compound production split incl. the mixed cells of the training compounds; out runs_final_unified.
    PaiNN lambda 0.3 + harmonic-residual energy (best energy, best mixed r_disp), PaiNN lambda 0.3 (best point/
    planar r_disp) and DefiNet (positions only) as ensemble partners."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    F = " --bs 4 --epochs 250 --unified --mixed train --final --save-preds --split-seed 0"
    R = NS + " " + EXT + " --tag dfrecipe" + F + " --energy-weight 0.3"
    D = EXT + " --ext-feats --ext-force --std-inputs" + F
    c = [("painn", "eqvF", R + " --e-harm"), ("painn", "eqvF", R), ("definet", "eqv", D)]
    return [cmd("joint", m, V, h, "mse", seed, group="disp-final-unified", out="runs_final_unified", extra=x)
            for m, h, x in c for seed in (0, 1)]


def exact():
    """Exact point energy labels (--e-exact, data/superset_point_exact.pt): the recommended production PaiNN
    (unified, mixed-trained, lambda 0.3 + harmonic-residual energy) 3 splits x 2 seeds (compare with runs_unified,
    rescored on the exact labels) + the 2 production members (--final). Out runs_exact."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    R = NS + " " + EXT + " --tag dfrecipe --bs 4 --epochs 250 --unified --mixed train --energy-weight 0.3 --e-harm --e-exact"
    study = _p3([("joint", "painn", V, "eqvF", R)], g="disp-exact", out="runs_exact")
    fin = [cmd("joint", "painn", V, "eqvF", "mse", seed, group="disp-exact", out="runs_exact",
               extra=R + " --final --save-preds --split-seed 0") for seed in (0, 1)]
    return study + fin


def separate():
    """Separate vs unified heads: the unified, mixed-trained PaiNN as a position-only model and as an energy-only
    model (plain head; and with the harmonic term -1/2 F0.d of its own, unsupervised d as a learned feature).
    Compare with the multi-task runs in runs_unified (ew0.3, ew0.3 + e-harm). Out runs_separate."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    R = NS + " " + EXT + " --tag dfrecipe --bs 4 --epochs 250 --unified --mixed train"
    c = [("joint", "painn", V, "eqvF", R),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 1 --pos-weight 0"),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 1 --pos-weight 0 --e-harm")]
    return _p3(c, g="disp-separate", out="runs_separate")


def energy():
    """Energy as a second output of the joint PaiNN: energy only / multi-task (3 weights) / + F0.d coupling.
    Position-only reference = runs_joint (same view, recipe, splits, seeds)."""
    V = "rt=8.9,buf=4,edge=k3,feat=base+cyc+force,stable=1"
    R = NS + " " + EXT + " --tag dfrecipe --bs 4 --epochs 250"
    c = [("joint", "painn", V, "eqvF", R + " --energy-weight 1 --pos-weight 0"),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 1"),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 0.3"),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 3"),
         ("joint", "painn", V, "eqvF", R + " --energy-weight 1 --e-coupling")]
    return _p3(c, g="disp-energy", out="runs_energy")


# ---- external benchmarks (the authors' datasets and splits; bench/README in RESULTS) ----
BENCH_EPOCHS = {"definet_low": 100, "definet_high": 300, "c2db": 300, "vdW": 300, "jarvis": 60, "xmno": 60, "mp": 40}


def bench_confs(ds):
    """4 arms x 3 seeds on one dataset: the authors' model as published, + our inputs, our PaiNN, our PaiNN
    without GRACE inputs. Defect sets use DefiNet, bulk sets (cell relaxes) E3Relax."""
    ext = "definet" if ds.startswith("definet") else "e3relax"
    ep = BENCH_EPOCHS[ds]
    bs = "--bs 8" if ds.startswith("definet") else "--bs 32"  # bulk sets: ~10-30 atoms per cell
    arms = [(ext, "eqv", EXT),
            (ext, "eqv", EXT + " --ext-feats --ext-force --std-inputs"),
            ("painn", "eqvF", NS + " --zemb"),
            ("painn", "eqv", NS + " --zemb --bench-feats none"),
            ("painn", "eqvF", NS + " --zemb " + EXT + " --tag dfrecipe")]  # best recipe of the ablation
    lines = []
    for seed in (0, 1, 2):
        for m, head, extra in arms:
            lines.append(cmd(f"bench:{ds}", m, "", head, "mse", seed, epochs=ep, group="disp-bench", out="runs_bench",
                             extra=f"--save-preds --eval-every 2 {bs} {extra}".strip()))
    return lines


def bench():
    import glob as _g
    names = [os.path.basename(f)[:-3] for f in sorted(_g.glob(os.path.join(HERE, "bench_data", "views", "*.pt")))]
    names = [n for n in names if n in BENCH_EPOCHS and n not in os.environ.get("BENCH_SKIP", "").split(",")]
    print("datasets:", names)
    return [l for n in names for l in bench_confs(n)]


if __name__ == "__main__":
    which = sys.argv[1]
    out = globals()[which]()
    if isinstance(out, tuple):  # (heavy, light): 2 heavy runs per GPU, 4 light ones
        heavy, light = out
        lines = [" ;; ".join(heavy[i:i + 2]) for i in range(0, len(heavy), 2)] + \
                [" ;; ".join(light[i:i + 4]) for i in range(0, len(light), 4)]
        path = os.path.join(HERE, f"tasks_{which}.txt")
        open(path, "w").write("\n".join(lines) + "\n")
        print(f"{path}: {len(heavy) + len(light)} runs in {len(lines)} tasks")
    else:
        per = {"ext_planar": 3, "ext_point": 1, "abl_planar": 3, "abl_point": 1, "bench": 1, "joint": 1, "final": 1, "final_mt": 1, "unified": 1, "final_unified": 1, "separate": 1, "exact": 1, "energy": 1}.get(which, 2 if which.endswith("h") else 4)
        write(which, out, per_line=per)
