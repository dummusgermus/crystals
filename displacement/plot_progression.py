"""Step-by-step improvement (val / test r_disp, mean of 3 compound splits x 2 seeds; numbers from
analysis_phase2/3/4 and ensemble.py, see RESULTS.md)."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = {"point": [("d = c·F0 (force × const)", .634, .623), ("CGCNN, k3 shells, base+cycles", .163, .257),
               ("+ force projections in head", .141, .233), ("+ train on stable only", .123, .202),
               ("+ LayerNorm, std. inputs (robust)", .118, .209), ("PaiNN (equivariant), cluster", .106, .195),
               ("PaiNN, full periodic cell", .106, .191), ("2-seed ensemble", .100, .173)],
     "planar": [("d = c·F0 (force × const)", .684, .715), ("CGCNN, k3 shells, base+cycles", .286, .314),
                ("+ force projections in head", .186, .219), ("+ train on stable only", .177, .202),
                ("PaiNN, force as start vector", .160, .171), ("+ norms, std. inputs, no cycles", .150, .169),
                ("+ lr 5e-4", .138, .150), ("2-seed ensemble", .133, .142)]}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, (dom, rows) in zip(axs, D.items()):
    y = range(len(rows))
    ax.barh([i - .2 for i in y], [r[1] for r in rows], .38, color="#9db8e8", label="validation")
    ax.barh([i + .2 for i in y], [r[2] for r in rows], .38, color="#2a6fdb", label="test (unseen compounds)")
    for i, r in enumerate(rows):
        ax.text(r[2] + .01, i + .2, f"{r[2]:.3f}", va="center", fontsize=8)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, .8); ax.set_xlabel("r_disp = |error| / |true d|  (median per structure; 1 = no motion)")
    ax.set_title(f"{dom} defects"); ax.grid(axis="x", alpha=.3)
axs[0].legend(loc="lower right")
fig.suptitle("Displacement prediction: what each step buys (3 compound splits × 2 seeds)")
fig.tight_layout(); out = os.path.join(HERE, "progression.png"); fig.savefig(out, dpi=130); print(out)
