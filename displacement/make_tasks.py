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


def write(name, lines, per_line=4):
    path = os.path.join(HERE, f"tasks_{name}.txt")
    grouped = [" ;; ".join(lines[i:i + per_line]) for i in range(0, len(lines), per_line)]
    with open(path, "w") as fh:
        fh.write("\n".join(grouped) + "\n")
    print(f"{path}: {len(lines)} runs in {len(grouped)} tasks")


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
        write(which, out)
