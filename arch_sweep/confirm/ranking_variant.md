# Ranking of `variant` vs. `base` (144 runs)

Paired test deltas (same split + seed), checkpoint = best val R_tot. Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.

### All settings

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | L4 | -4.60 | -2.6 | 4/4 | 3/4 | 5.42e-08 | 0.119 |
| 2 | L4-rbf16 | -4.19 | -11.5 | 4/4 | 3/4 | 1.99e-05 | 6.83e-05 |
| 3 | L3 | -3.34 | -3.3 | 4/4 | 3/4 | 0.000496 | 0.00429 |

### CGCNN only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | L4 | -4.60 | -2.6 | 4/4 | 3/4 | 5.42e-08 | 0.119 |
| 2 | L4-rbf16 | -4.19 | -11.5 | 4/4 | 3/4 | 1.99e-05 | 6.83e-05 |
| 3 | L3 | -3.34 | -3.3 | 4/4 | 3/4 | 0.000496 | 0.00429 |

### Transformer only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|

### Mean test R_tot (%) / MAE (meV/atom) per setting

| setting | L3 | L4 | L4-rbf16 | base |
|---|---|---|---|---|
| random/point/cgcnn | 21.4 / 8.4 | 20.1 / 8.8 | 20.7 / 7.8 | 24.4 / 9.0 |
| random/point/transformer | nan / nan | nan / nan | nan / nan | nan / nan |
| random/planar/cgcnn | 15.7 / 6.0 | 13.8 / 5.8 | 12.0 / 5.3 | 17.8 / 6.3 |
| random/planar/transformer | nan / nan | nan / nan | nan / nan | nan / nan |
| chem/point/cgcnn | 22.1 / 9.1 | 22.9 / 8.9 | 23.0 / 8.5 | 27.0 / 10.2 |
| chem/point/transformer | nan / nan | nan / nan | nan / nan | nan / nan |
| chem/planar/cgcnn | 35.1 / 23.2 | 32.4 / 23.8 | 35.2 / 21.4 | 38.4 / 21.3 |
| chem/planar/transformer | nan / nan | nan / nan | nan / nan | nan / nan |
