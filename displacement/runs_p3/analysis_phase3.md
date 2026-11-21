# Displacement study: phase3

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force stable-only | eqv | mse | 6 | 0.103 ± 0.01 | 0.174 | -54.834 | 0.0091 | 1.00 | 0.296 | – | 0.0220 | 268 | 55 |
| 2 | rt8.9/buf4 k3 base+force | eqv | mse | 6 | 0.190 ± 0.10 | 0.268 | -7642341693993121096925184.000 | 1095376005.0365 | 0.99 | 0.373 | – | 1.8216 | 172 | 59 |

## point_full · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9 k3 base+force stable-only | eqv | mse | 6 | 0.129 ± 0.06 | 0.200 | -123.175 | 0.0108 | 0.99 | 0.295 | – | 0.0551 | 258 | 171 |
| 2 | rt8.9 k3 base+force | eqv | mse | 6 | 0.179 ± 0.04 | 0.240 | -20.897 | 0.0105 | 0.99 | 0.328 | – | 0.0476 | 122 | 181 |
| 3 | rt8.9 k2 base+force | eqv | mse | 6 | 0.196 ± 0.08 | 0.255 | 0.529 | 0.0092 | 0.99 | 0.359 | – | 0.0552 | 143 | 120 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force stable-only | eqvF | mse | 6 | 0.160 ± 0.04 | 0.171 | 0.395 | 0.0104 | 1.00 | 0.183 | 0.0359 | – | 191 | 52 |
| 2 | k3 base+force | eqvF | mse | 6 | 0.169 ± 0.04 | 0.182 | 0.364 | 0.0108 | 1.00 | 0.190 | 0.0384 | – | 173 | 53 |
| 3 | k3 base+cyc+force | eqvF | mse | 6 | 0.184 ± 0.06 | 0.199 | 0.369 | 0.0111 | 1.00 | 0.209 | 0.0395 | – | 132 | 73 |
| 4 | k3 base+cyc+force [wd1e-3] | eqvF | mse | 6 | 0.192 ± 0.04 | 0.202 | 0.342 | 0.0114 | 1.00 | 0.215 | 0.0466 | – | 127 | 53 |
| 5 | k3 base+cyc+force | eqv | mse | 6 | 0.194 ± 0.06 | 0.207 | 0.376 | 0.0112 | 1.00 | 0.218 | 0.0464 | – | 128 | 72 |
| 6 | k3 base+cyc+force [l6] | eqvF | mse | 6 | 0.337 ± 0.05 | 0.335 | 0.354 | 0.0128 | 1.00 | 0.348 | 0.0323 | – | 8 | 55 |
