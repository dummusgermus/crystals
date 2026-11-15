# Production retrain: old vs. new (delivery validation set)

New = mean over seeds [min - max]. Val set was also used for checkpoint selection (both old and new).


## point_cgcnn (3 seeds; target mode old=full, new=full)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 12.02 | 5.58 [4.67 - 6.71] | -53.6 % |
| R_tot graph-sum (%) | 30.34 | 12.07 [6.96 - 16.47] | -60.2 % |
| R_tot full-cell (%) | 34.79 | 11.38 [10.68 - 11.77] | -67.3 % |
| |dE| graph-sum (meV) | 534.99 | 274.90 [185.79 - 401.04] | -48.6 % |

## planar_cgcnn (3 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 26.53 | 6.05 [5.96 - 6.11] | -77.2 % |
| R_tot graph-sum (%) | 32.77 | 6.96 [6.90 - 6.99] | -78.8 % |
| R_tot full-cell (%) | 32.77 | 6.96 [6.90 - 6.99] | -78.8 % |
| |dE| graph-sum (meV) | 859.15 | 62.72 [55.11 - 69.34] | -92.7 % |

## point_transformer (3 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 7.28 | 6.94 [6.73 - 7.10] | -4.7 % |
| R_tot graph-sum (%) | 15.98 | 9.88 [8.66 - 11.49] | -38.2 % |
| R_tot full-cell (%) | 21.25 | 18.19 [16.51 - 19.11] | -14.4 % |
| |dE| graph-sum (meV) | 253.52 | 306.81 [221.29 - 472.23] | +21.0 % |

## planar_transformer (3 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 32.86 | 12.76 [12.61 - 12.88] | -61.2 % |
| R_tot graph-sum (%) | 46.46 | 7.40 [6.65 - 8.00] | -84.1 % |
| R_tot full-cell (%) | 46.46 | 7.40 [6.65 - 8.00] | -84.1 % |
| |dE| graph-sum (meV) | 1033.82 | 61.36 [54.51 - 72.03] | -94.1 % |
