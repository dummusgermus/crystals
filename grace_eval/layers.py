"""Layer analysis of hexagonal Laves cells (c axis = z)."""
import numpy as np

def layers(pos, types, tol=0.25):
    z = pos[:, 2]; order = np.argsort(z)
    out = []; cur = [order[0]]
    for i in order[1:]:
        if z[i] - z[cur[-1]] > tol:
            out.append(cur); cur = [i]
        else:
            cur.append(i)
    out.append(cur)
    return [(float(np.mean(z[l])), "".join(sorted("AB"[int(types[i]) - 1] for i in l)), sorted(l)) for l in out]


def units(pos, types, tol=0.25):
    """Group layers into Laves units: BBB kagome layer + following A-B-A triple layer.
    Returns list of dicts with triple-layer atom indices and the lateral class of the middle B."""
    L = layers(pos, types, tol)
    start = next(i for i, l in enumerate(L) if l[1] == "BBB")
    L = L[start:] + L[:start]
    if len(L) % 4:
        raise ValueError("layer count not a multiple of 4")
    kag = [p for l in L if l[1] == "BBB" for p in l[2]]
    k0 = pos[L[0][2]]
    d = [np.linalg.norm(k0[i, :2] - k0[j, :2]) for i in range(3) for j in range(i + 1, 3)]
    a = 2 * min(d)
    a1 = np.array([a, 0.0]); a2 = np.array([-a / 2, a * np.sqrt(3) / 2])
    inv = np.linalg.inv(np.stack([a1, a2]).T)
    out = []
    for u in range(len(L) // 4):
        kl, l1, l2, l3 = L[4 * u: 4 * u + 4]
        assert kl[1] == "BBB" and l1[1] == "A" and l2[1] == "B" and l3[1] == "A", [kl[1], l1[1], l2[1], l3[1]]
        f = inv @ pos[l2[2][0], :2]
        cls = int(round(3 * (f[0] - f[1]))) % 3
        fa = inv @ pos[l1[2][0], :2]
        cls_a = int(round(3 * (fa[0] - fa[1]))) % 3
        out.append({"triple": l1[2] + l2[2] + l3[2], "kagome": kl[2], "cls": cls, "cls_a": cls_a})
    return out


def hc_sequence(us):
    """'h' if the neighbouring units of a unit are laterally identical, else 'c' (periodic)."""
    n = len(us)
    return "".join("h" if us[(i - 1) % n]["cls"] == us[(i + 1) % n]["cls"] else "c" for i in range(n))
