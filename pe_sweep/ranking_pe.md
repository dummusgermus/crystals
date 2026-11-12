# Ranking of `pe` vs. `none` (204 runs)

Paired test deltas (same split + seed), checkpoint = best val R_tot. Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.

### All settings

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | lap8+rwse16 | -3.46 | +3.0 | 5/8 | 4/8 | 0.203 | 0.292 |
| 2 | rwse8 | -2.09 | +17.1 | 5/8 | 3/8 | 0.219 | 0.698 |
| 3 | lap8 | +0.51 | +5.6 | 3/8 | 2/8 | 0.772 | 0.616 |
| 4 | lap16 | +1.07 | +13.5 | 3/8 | 2/8 | 0.55 | 0.849 |
| 5 | rwse16 | +1.62 | +9.7 | 4/8 | 3/8 | 0.708 | 0.906 |
| 6 | lap8+rwse16+spd+geo | +18.71 | +56.2 | 0/4 | 0/4 | 0.987 | 1 |
| 7 | spd | +19.36 | +61.7 | 1/4 | 0/4 | 0.987 | 1 |
| 8 | rwse16+spd+geo | +24.17 | +70.5 | 0/4 | 0/4 | 0.99 | 1 |
| 9 | spd+geo | +28.15 | +59.4 | 0/4 | 0/4 | 0.994 | 1 |
| 10 | geo | +38.43 | +51.4 | 0/4 | 0/4 | 0.998 | 1 |

### CGCNN only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | lap8+rwse16 | -4.91 | -3.2 | 3/4 | 2/4 | 0.0549 | 0.311 |
| 2 | lap16 | -3.41 | +9.3 | 2/4 | 1/4 | 0.17 | 0.741 |
| 3 | rwse16 | -2.89 | +13.7 | 3/4 | 2/4 | 0.367 | 0.715 |
| 4 | rwse8 | -1.61 | +24.9 | 2/4 | 2/4 | 0.455 | 0.765 |
| 5 | lap8 | +2.37 | +0.3 | 1/4 | 1/4 | 0.765 | 0.883 |

### Transformer only

| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |
|---|---|---|---|---|---|---|---|
| 1 | rwse8 | -2.56 | +9.2 | 3/4 | 1/4 | 0.117 | 0.604 |
| 2 | lap8+rwse16 | -2.02 | +9.2 | 2/4 | 2/4 | 0.633 | 0.425 |
| 3 | lap8 | -1.35 | +11.0 | 2/4 | 1/4 | 0.661 | 0.425 |
| 4 | lap16 | +5.56 | +17.8 | 1/4 | 1/4 | 0.883 | 0.788 |
| 5 | rwse16 | +6.13 | +5.7 | 1/4 | 1/4 | 0.83 | 0.883 |
| 6 | lap8+rwse16+spd+geo | +18.71 | +56.2 | 0/4 | 0/4 | 0.987 | 1 |
| 7 | spd | +19.36 | +61.7 | 1/4 | 0/4 | 0.987 | 1 |
| 8 | rwse16+spd+geo | +24.17 | +70.5 | 0/4 | 0/4 | 0.99 | 1 |
| 9 | spd+geo | +28.15 | +59.4 | 0/4 | 0/4 | 0.994 | 1 |
| 10 | geo | +38.43 | +51.4 | 0/4 | 0/4 | 0.998 | 1 |

### Mean test R_tot (%) / MAE (meV/atom) per setting

| setting | geo | lap16 | lap8 | lap8+rwse16 | lap8+rwse16+spd+geo | none | rwse16 | rwse16+spd+geo | rwse8 | spd | spd+geo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random/point/cgcnn | nan / nan | 30.6 / 12.1 | 38.5 / 12.5 | 26.8 / 12.0 | nan / nan | 29.8 / 11.6 | 28.2 / 11.1 | nan / nan | 27.9 / 11.0 | nan / nan | nan / nan |
| random/point/transformer | 50.9 / 16.7 | 42.9 / 15.6 | 35.8 / 13.9 | 25.8 / 12.1 | 50.7 / 16.8 | 21.8 / 12.5 | 28.2 / 14.3 | 47.3 / 16.3 | 36.4 / 13.7 | 44.0 / 16.2 | 43.1 / 16.7 |
| random/planar/cgcnn | nan / nan | 41.8 / 18.9 | 42.7 / 14.7 | 40.4 / 12.0 | nan / nan | 38.0 / 12.0 | 39.7 / 18.4 | nan / nan | 43.9 / 27.3 | nan / nan | nan / nan |
| random/planar/transformer | 77.4 / 33.9 | 38.1 / 19.5 | 37.8 / 18.3 | 39.9 / 22.7 | 57.7 / 35.8 | 34.3 / 16.3 | 33.0 / 14.2 | 75.8 / 44.6 | 32.6 / 21.5 | 64.5 / 36.5 | 68.7 / 36.1 |
| chem/point/cgcnn | nan / nan | 33.4 / 9.9 | 33.3 / 9.6 | 27.3 / 9.2 | nan / nan | 41.3 / 9.5 | 30.7 / 9.4 | nan / nan | 30.4 / 10.3 | nan / nan | nan / nan |
| chem/point/transformer | 50.7 / 13.1 | 36.8 / 10.4 | 25.0 / 8.3 | 26.8 / 8.4 | 42.1 / 12.7 | 41.1 / 10.7 | 59.9 / 12.4 | 41.3 / 12.7 | 20.9 / 9.1 | 35.3 / 12.9 | 42.7 / 13.1 |
| chem/planar/cgcnn | nan / nan | 42.6 / 22.7 | 57.0 / 22.2 | 47.9 / 27.7 | nan / nan | 53.0 / 31.8 | 51.9 / 34.1 | nan / nan | 53.4 / 22.2 | nan / nan | nan / nan |
| chem/planar/transformer | 117.7 / 33.5 | 47.3 / 30.5 | 38.9 / 33.5 | 42.4 / 29.1 | 67.3 / 36.1 | 45.7 / 23.6 | 46.4 / 24.9 | 75.3 / 37.6 | 42.8 / 26.0 | 76.6 / 40.9 | 101.0 / 37.8 |
