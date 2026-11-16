# Production retrain: old vs. new (delivery validation set)

New = mean over seeds [min - max]. Val set was also used for checkpoint selection (both old and new).


## point_cgcnn (5 seeds; target mode old=full, new=full)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 12.02 | 5.29 [4.67 - 6.71] | -56.0 % |
| R_tot graph-sum (%) | 30.34 | 12.22 [6.96 - 16.47] | -59.7 % |
| R_tot full-cell (%) | 34.79 | 11.19 [10.68 - 11.77] | -67.8 % |
| |dE| graph-sum (meV) | 534.99 | 255.59 [185.79 - 401.04] | -52.2 % |

## planar_cgcnn (5 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 26.53 | 5.88 [5.19 - 6.11] | -77.9 % |
| R_tot graph-sum (%) | 32.77 | 6.72 [5.85 - 6.99] | -79.5 % |
| R_tot full-cell (%) | 32.77 | 6.72 [5.85 - 6.99] | -79.5 % |
| |dE| graph-sum (meV) | 859.15 | 58.97 [45.05 - 69.34] | -93.1 % |

## point_transformer (5 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 7.28 | 7.35 [6.73 - 8.28] | +1.0 % |
| R_tot graph-sum (%) | 15.98 | 9.62 [7.11 - 11.49] | -39.8 % |
| R_tot full-cell (%) | 21.25 | 18.94 [16.51 - 20.42] | -10.9 % |
| |dE| graph-sum (meV) | 253.52 | 408.64 [221.29 - 564.73] | +61.2 % |

## planar_transformer (5 seeds; target mode old=graph, new=graph)

| metric | old | new mean [min - max] | change |
|---|---|---|---|
| val MAE (meV/atom) | 32.86 | 12.87 [12.61 - 13.21] | -60.8 % |
| R_tot graph-sum (%) | 46.46 | 7.67 [6.65 - 8.14] | -83.5 % |
| R_tot full-cell (%) | 46.46 | 7.67 [6.65 - 8.14] | -83.5 % |
| |dE| graph-sum (meV) | 1033.82 | 66.22 [37.56 - 109.47] | -93.6 % |
