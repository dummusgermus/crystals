# Atom-type encoding ranking (vs. current type 1/2)

Mean paired change vs. type12 over 8 settings (2 splits × point/planar × CGCNN/Transformer). Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.

| rank | encoding | ΔR_tot (pp) | ΔMAE (%) | settings improved (R_tot) | p(R_tot) | p(MAE) | better than current |
|---|---|---|---|---|---|---|---|
| 1 | onehot_props (one-hot + descriptors + role) | -7.6 | -15.5 | 8/8 | 5.05e-05 | 0.000168 | yes |
| 2 | elem_props_role (descriptors + role) | -7.3 | -15.7 | 7/8 | 0.000302 | 1.65e-05 | yes |
| 3 | elem_onehot (element one-hot) | -7.1 | -7.3 | 6/8 | 0.00153 | 0.0225 | yes |
| 4 | pair_props_edge (own + partner descr. + edge feats) | -6.6 | -21.9 | 8/8 | 0.000149 | 1.6e-06 | yes |
| 5 | pair_props (own + partner descriptors) | -6.5 | -19.6 | 7/8 | 0.000497 | 7.99e-05 | yes |
| 6 | elem_props (element descriptors) | -5.7 | -11.8 | 7/8 | 0.00176 | 0.00531 | yes |
| 7 | role_onehot (role one-hot) | -0.5 | -8.1 | 4/8 | 0.504 | 0.000453 | no |
| 8 | none (no type) | +8.7 | +7.0 | 1/8 | 0.999 | 0.868 | no |

## Best encoding per setting (test R_tot median, %)

| split | setting | current | best | encoding | seed wins |
|---|---|---|---|---|---|
| random | Point · CGCNN | 31.8 | 26.9 | pair_props | 5/5 |
| random | Point · Transformer | 34.3 | 20.7 | elem_props | 4/5 |
| random | Planar · CGCNN | 45.0 | 38.0 | elem_onehot | 4/5 |
| random | Planar · Transformer | 46.5 | 31.1 | onehot_props | 5/5 |
| chem | Point · CGCNN | 31.2 | 24.9 | pair_props_edge | 4/5 |
| chem | Point · Transformer | 35.0 | 19.4 | onehot_props | 5/5 |
| chem | Planar · CGCNN | 56.9 | 47.2 | elem_onehot | 4/5 |
| chem | Planar · Transformer | 61.3 | 41.2 | elem_onehot | 4/5 |
