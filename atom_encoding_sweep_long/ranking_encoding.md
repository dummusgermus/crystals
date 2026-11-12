# Ranking of `encoding` vs. `type12` (96 runs)

Paired test deltas (same split + seed), checkpoint = best val R_tot. Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.

### All settings

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | pair_props_edge | -11.04 | -13.6 | 8/8 | 7/8 | 1.01e-05 | 0.00479 |
| 2 | elem_props_role | -9.01 | -17.5 | 7/8 | 7/8 | 7.48e-05 | 2.56e-06 |
| 3 | onehot_props | -7.97 | -16.1 | 6/8 | 7/8 | 0.00157 | 0.0082 |

### CGCNN only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | pair_props_edge | -7.94 | +4.3 | 4/4 | 3/4 | 0.00122 | 0.285 |
| 2 | elem_props_role | -5.47 | -17.2 | 3/4 | 3/4 | 0.00464 | 0.00171 |
| 3 | onehot_props | -2.16 | -1.7 | 2/4 | 3/4 | 0.151 | 0.396 |

### Transformer only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | pair_props_edge | -14.15 | -31.5 | 4/4 | 4/4 | 0.00342 | 0.000244 |
| 2 | onehot_props | -13.78 | -30.5 | 4/4 | 4/4 | 0.00244 | 0.00171 |
| 3 | elem_props_role | -12.55 | -17.8 | 4/4 | 4/4 | 0.00464 | 0.000244 |

### Mean test R_tot (%) / MAE (meV/atom) per setting

| setting | elem_props_role | onehot_props | pair_props_edge | type12 |
|---|---|---|---|---|
| random/point/cgcnn | 25.9 / 12.4 | 34.6 / 12.2 | 24.4 / 11.3 | 37.1 / 13.0 |
| random/point/transformer | 23.5 / 12.4 | 17.7 / 10.3 | 22.0 / 13.1 | 39.6 / 16.1 |
| random/planar/cgcnn | 38.6 / 10.7 | 36.0 / 19.7 | 44.6 / 35.0 | 45.1 / 20.7 |
| random/planar/transformer | 38.2 / 24.3 | 31.2 / 15.1 | 35.1 / 13.9 | 52.5 / 35.5 |
| chem/point/cgcnn | 24.7 / 8.3 | 32.3 / 9.6 | 24.1 / 9.2 | 30.0 / 9.9 |
| chem/point/transformer | 23.1 / 9.0 | 28.7 / 9.9 | 25.7 / 8.9 | 32.7 / 10.7 |
| chem/planar/cgcnn | 55.0 / 34.3 | 54.6 / 36.4 | 41.4 / 23.5 | 53.9 / 34.2 |
| chem/planar/transformer | 43.6 / 34.4 | 45.9 / 27.2 | 39.2 / 24.3 | 53.8 / 34.5 |
