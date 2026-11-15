# Plateau fix confirmation

Paired vs. current recipe (same split+seed). 'gated' = variant @ safer checkpoint vs. base @ best-val-R_tot (production). R_tot in %, MAE in meV/atom; Δ averaged over random+chem; wins = paired runs with lower test R_tot.

| model / domain | variant | base R_tot / MAE | variant R_tot / MAE (gated) | ΔR_tot pp | ΔMAE % | wins R_tot | wins MAE | p(R_tot) | p(MAE) | ΔR_tot pp (both ckpt_rtot) |
|---|---|---|---|---|---|---|---|---|---|---|
| cgcnn / planar | bn | 49.8 / 22.9 | 86.9 / 24.3 | +37.1 | +7.2 | 2/12 | 6/12 | 0.999 | 0.741 | +11.9 |
| cgcnn / planar | clip | 49.8 / 22.9 | 45.8 / 17.9 | -4.1 | -21.6 | 8/12 | 8/12 | 0.19 | 0.0261 | -5.1 |
| cgcnn / planar | combo | 49.8 / 22.9 | 32.3 / 11.9 | -17.5 | -47.1 | 12/12 | 12/12 | 0.000244 | 0.000244 | -19.0 |
| cgcnn / planar | combo-rob | 49.8 / 22.9 | 26.9 / 11.8 | -23.0 | -49.1 | 12/12 | 11/12 | 0.000244 | 0.000488 | -23.0 |
| cgcnn / planar | cos | 49.8 / 22.9 | 59.0 / 27.3 | +9.2 | +40.4 | 3/12 | 4/12 | 0.954 | 0.849 | +3.3 |
| cgcnn / planar | lr1e-3 | 49.8 / 22.9 | 47.7 / 19.8 | -2.1 | -20.3 | 7/12 | 8/12 | 0.235 | 0.0461 | -6.7 |
| cgcnn / planar | lr5e-4 | 49.8 / 22.9 | 45.9 / 16.2 | -3.9 | -29.8 | 8/12 | 10/12 | 0.133 | 0.00122 | -5.1 |
| cgcnn / planar | msewu | 49.8 / 22.9 | 62.6 / 18.4 | +12.7 | -2.7 | 3/12 | 8/12 | 0.968 | 0.212 | +12.8 |
| cgcnn / planar | pat30 | 49.8 / 22.9 | 56.6 / 24.7 | +6.7 | +17.5 | 5/12 | 4/12 | 0.883 | 0.849 | +4.3 |
| cgcnn / planar | rob | 49.8 / 22.9 | 51.0 / 22.0 | +1.2 | -7.8 | 5/12 | 7/12 | 0.765 | 0.133 | -0.2 |
| cgcnn / point | combo | 28.1 / 9.6 | 25.4 / 9.0 | -2.7 | -7.1 | 9/12 | 10/12 | 0.0757 | 0.0261 | -4.7 |
| cgcnn / point | combo-rob | 28.1 / 9.6 | 23.7 / 8.6 | -4.4 | -12.0 | 9/12 | 11/12 | 0.0105 | 0.00171 | -5.2 |
| transformer / planar | rob | 42.5 / 23.0 | 47.1 / 20.8 | +4.6 | -16.1 | 4/12 | 8/12 | 0.898 | 0.117 | -2.9 |
| transformer / point | rob | 32.4 / 10.2 | 20.4 / 10.2 | -12.0 | +4.8 | 9/12 | 8/12 | 0.00342 | 0.339 | -12.5 |
