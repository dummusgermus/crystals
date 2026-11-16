# Delivery val set: old vs. retrained single seeds vs. 3- and 5-seed mean ensembles

| model | method | MAE (meV) | R_tot graph % | R_tot full % | |dE| graph (eV) |
|---|---|---|---|---|---|
| point_cgcnn | old | 12.02 | 30.34 | 34.79 | 0.535 |
| point_cgcnn | new s0 | 4.67 | 6.96 | 10.68 | 0.186 |
| point_cgcnn | new s1 | 6.71 | 16.47 | 11.70 | 0.401 |
| point_cgcnn | new s2 | 5.36 | 12.79 | 11.77 | 0.238 |
| point_cgcnn | new s3 | 4.88 | 14.18 | 10.83 | 0.224 |
| point_cgcnn | new s4 | 4.80 | 10.69 | 10.95 | 0.230 |
| point_cgcnn | new ensemble (3) | 5.21 | 9.37 | 11.97 | 0.258 |
| point_cgcnn | new ensemble (5) | 4.87 | 8.63 | 12.41 | 0.232 |
| planar_cgcnn | old | 26.53 | 32.77 | 32.77 | 0.859 |
| planar_cgcnn | new s0 | 6.11 | 6.90 | 6.90 | 0.069 |
| planar_cgcnn | new s1 | 5.96 | 6.99 | 6.99 | 0.055 |
| planar_cgcnn | new s2 | 6.08 | 6.99 | 6.99 | 0.064 |
| planar_cgcnn | new s3 | 5.19 | 5.85 | 5.85 | 0.045 |
| planar_cgcnn | new s4 | 6.04 | 6.86 | 6.86 | 0.062 |
| planar_cgcnn | new ensemble (3) | 5.70 | 5.92 | 5.92 | 0.057 |
| planar_cgcnn | new ensemble (5) | 5.44 | 5.62 | 5.62 | 0.053 |
| point_transformer | old | 7.28 | 15.98 | 21.25 | 0.254 |
| point_transformer | new s0 | 6.99 | 9.49 | 18.95 | 0.221 |
| point_transformer | new s1 | 7.10 | 11.49 | 16.51 | 0.227 |
| point_transformer | new s2 | 6.73 | 8.66 | 19.11 | 0.472 |
| point_transformer | new s3 | 8.28 | 11.35 | 19.71 | 0.565 |
| point_transformer | new s4 | 7.65 | 7.11 | 20.42 | 0.558 |
| point_transformer | new ensemble (3) | 6.60 | 13.16 | 20.61 | 0.289 |
| point_transformer | new ensemble (5) | 6.98 | 11.70 | 20.54 | 0.392 |
| planar_transformer | old | 32.86 | 46.46 | 46.46 | 1.034 |
| planar_transformer | new s0 | 12.79 | 7.55 | 7.55 | 0.055 |
| planar_transformer | new s1 | 12.61 | 8.00 | 8.00 | 0.058 |
| planar_transformer | new s2 | 12.88 | 6.65 | 6.65 | 0.072 |
| planar_transformer | new s3 | 12.89 | 8.14 | 8.14 | 0.038 |
| planar_transformer | new s4 | 13.21 | 8.03 | 8.03 | 0.109 |
| planar_transformer | new ensemble (3) | 12.54 | 6.54 | 6.54 | 0.056 |
| planar_transformer | new ensemble (5) | 12.59 | 6.58 | 6.58 | 0.062 |
