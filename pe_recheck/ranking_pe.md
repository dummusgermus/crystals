# Ranking of `pe` vs. `none` (48 runs)

Paired test deltas (same split + seed), checkpoint = best val R_tot. Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.

### All settings

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | lap8+rwse16 | -1.03 | +4.8 | 4/4 | 2/4 | 0.104 | 0.678 |

### CGCNN only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | lap8+rwse16 | -1.03 | +4.8 | 4/4 | 2/4 | 0.104 | 0.678 |

### Transformer only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|

### Mean test R_tot (%) / MAE (meV/atom) per setting

| setting | lap8+rwse16 | none |
|---|---|---|
| random/point/cgcnn | 20.2 / 7.9 | 22.9 / 7.5 |
| random/point/transformer | nan / nan | nan / nan |
| random/planar/cgcnn | 11.4 / 5.5 | 12.0 / 5.6 |
| random/planar/transformer | nan / nan | nan / nan |
| chem/point/cgcnn | 21.8 / 6.8 | 22.5 / 7.1 |
| chem/point/transformer | nan / nan | nan / nan |
| chem/planar/cgcnn | 33.3 / 18.4 | 33.4 / 15.3 |
| chem/planar/transformer | nan / nan | nan / nan |
