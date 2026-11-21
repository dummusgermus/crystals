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
        per = {"ext_planar": 3, "ext_point": 1, "abl_planar": 3, "abl_point": 1}.get(which, 2 if which.endswith("h") else 4)
        write(which, out, per_line=per)
