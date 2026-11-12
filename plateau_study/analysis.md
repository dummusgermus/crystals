# Plateau study: planar CGCNN, pair_props_edge, 2000 epochs

Stuck = val MAE never below 20 meV/atom. Δ = paired vs. base (same split+seed), mean over random+chem; p = one-sided Wilcoxon (variant better).

| variant | stuck | late relapse | median escape ep | rtot: R_tot / MAE | gated: R_tot / MAE | mae: R_tot / MAE | ΔR_tot gated (pp) | ΔMAE gated (%) | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 2/12 | 0/12 | 20.5 | 49.8 / 22.9 | 49.3 / 20.6 | 53.0 / 22.7 | -0.6 | -12.1 | 0.0721 | 0.235 |
| bn | 2/12 | 2/12 | 16.5 | 61.7 / 34.3 | 86.9 / 24.3 | 132.8 / 26.8 | +37.1 | +3.4 | 0.999 | 0.689 |
| clip | 1/12 | 1/12 | 12.0 | 44.7 / 16.8 | 45.8 / 17.9 | 51.0 / 20.1 | -4.1 | -26.3 | 0.19 | 0.0261 |
| combo | 1/12 | 1/12 | 16.0 | 30.9 / 11.8 | 32.3 / 11.9 | 36.6 / 11.9 | -17.5 | -50.9 | 0.000244 | 0.000244 |
| combo-rob | 1/12 | 0/12 | 13.0 | 26.8 / 11.8 | 26.9 / 11.8 | 31.5 / 13.6 | -23.0 | -51.4 | 0.000244 | 0.000488 |
| cos | 4/12 | 3/12 | 13.0 | 53.1 / 28.1 | 59.0 / 27.3 | 80.3 / 28.5 | +9.2 | +22.5 | 0.954 | 0.83 |
| lr1e-3 | 1/12 | 2/12 | 15.0 | 43.2 / 17.2 | 47.7 / 19.8 | 51.8 / 19.8 | -2.1 | -19.8 | 0.235 | 0.0461 |
| lr5e-4 | 1/12 | 1/12 | 12.0 | 44.7 / 16.4 | 45.9 / 16.2 | 51.2 / 17.9 | -3.9 | -33.6 | 0.133 | 0.00122 |
| msewu | 4/12 | 0/12 | 8.5 | 62.7 / 18.7 | 62.6 / 18.4 | 88.3 / 18.8 | +12.7 | -19.4 | 0.968 | 0.0881 |
| pat30 | 4/12 | 1/12 | 7.5 | 54.1 / 26.5 | 56.6 / 24.7 | 60.3 / 25.1 | +6.7 | +7.1 | 0.883 | 0.849 |
| rob | 1/12 | 1/12 | 18.0 | 49.6 / 26.0 | 51.0 / 22.0 | 51.0 / 21.3 | +1.2 | -8.5 | 0.765 | 0.151 |

Δ columns compare variant @ gated checkpoint vs. the CURRENT production setup (base @ best-val-R_tot).

## Per split (test R_tot % / MAE meV at gated checkpoint, stuck count)

| variant | random | chem |
|---|---|---|
| base | 44.6 / 14.4 (stuck 1/6) | 53.9 / 26.8 (stuck 1/6) |
| bn | 99.9 / 16.8 (stuck 0/6) | 73.9 / 31.8 (stuck 2/6) |
| clip | 35.7 / 9.0 (stuck 0/6) | 55.8 / 26.8 (stuck 1/6) |
| combo | 20.7 / 6.2 (stuck 0/6) | 44.0 / 17.5 (stuck 1/6) |
| combo-rob | 18.3 / 6.1 (stuck 0/6) | 35.4 / 17.5 (stuck 1/6) |
| cos | 58.7 / 27.4 (stuck 4/6) | 59.3 / 27.3 (stuck 0/6) |
| lr1e-3 | 36.5 / 8.1 (stuck 0/6) | 58.9 / 31.6 (stuck 1/6) |
| lr5e-4 | 32.8 / 7.6 (stuck 0/6) | 59.0 / 24.9 (stuck 1/6) |
| msewu | 53.7 / 15.7 (stuck 2/6) | 71.4 / 21.1 (stuck 2/6) |
| pat30 | 54.7 / 19.9 (stuck 2/6) | 58.4 / 29.5 (stuck 2/6) |
| rob | 48.5 / 12.3 (stuck 0/6) | 53.4 / 31.7 (stuck 1/6) |
