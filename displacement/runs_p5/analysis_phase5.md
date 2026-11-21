# Displacement study: phase5

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in [h256] | eqv | mse | 5 | 0.092 ± 0.01 | 0.191 | 0.714 | 0.0076 | 1.00 | 0.290 | – | 0.2048 | 297 | 66 |
| 2 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in [l5] | eqv | mse | 5 | 0.103 ± 0.00 | 0.205 | 0.680 | 0.0085 | 1.00 | 0.340 | – | 0.1406 | 308 | 59 |
| 3 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in [lr5e-4] | eqv | mse | 6 | 0.105 ± 0.00 | 0.186 | 0.734 | 0.0075 | 1.00 | 0.299 | – | 0.0632 | 277 | 53 |
| 4 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in [l3] | eqv | mse | 6 | 0.107 ± 0.01 | 0.204 | 0.712 | 0.0073 | 1.00 | 0.307 | – | 0.0644 | 309 | 48 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+force stable-only LN std-in [lr5e-4] | eqvF | mse | 6 | 0.138 ± 0.02 | 0.150 | 0.281 | 0.0112 | 1.00 | 0.163 | 0.0308 | – | 238 | 30 |
| 2 | k3 base+force stable-only LN std-in [l5] | eqvF | mse | 6 | 0.145 ± 0.02 | 0.163 | 0.296 | 0.0111 | 1.00 | 0.176 | 0.0295 | – | 219 | 31 |
| 3 | k3 base+force stable-only LN std-in [h256] | eqvF | mse | 6 | 0.150 ± 0.02 | 0.180 | 0.313 | 0.0113 | 1.00 | 0.192 | 0.0291 | – | 241 | 29 |
| 4 | k3 base+force stable-only LN std-in [l3] | eqvF | mse | 6 | 0.155 ± 0.02 | 0.173 | 0.338 | 0.0110 | 1.00 | 0.184 | 0.0315 | – | 205 | 29 |
