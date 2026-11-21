# Displacement study: phase4

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in | eqv+F | mse | 6 | 0.116 ± 0.02 | 0.213 | 0.742 | 0.0074 | 1.00 | 0.310 | – | 0.0139 | 280 | 42 |

## point · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in | eqv | mse | 6 | 0.104 ± 0.01 | 0.185 | 0.746 | 0.0071 | 1.00 | 0.307 | – | 0.0772 | 250 | 53 |
| 2 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in | eqv | mse | 5 | 0.107 ± 0.01 | 0.202 | 0.695 | 0.0078 | 1.00 | 0.296 | – | 0.0586 | 245 | 53 |

## point_full · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9 k3 base+force stable-only LN std-in | eqv+F | mse | 6 | 0.112 ± 0.01 | 0.191 | 0.751 | 0.0068 | 1.00 | 0.296 | – | 0.0092 | 272 | 129 |

## point_full · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9 k3 base+force stable-only LN std-in | eqv | mse | 5 | 0.096 ± 0.01 | 0.193 | 0.749 | 0.0075 | 1.00 | 0.337 | – | 0.0422 | 293 | 156 |
| 2 | rt8.9 k3 base+force stable-only LN std-in | eqv | mse | 5 | 0.109 ± 0.00 | 0.175 | 0.818 | 0.0062 | 1.00 | 0.295 | – | 0.0308 | 270 | 155 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · painn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+force stable-only LN std-in | eqvF | mse | 6 | 0.150 ± 0.02 | 0.169 | 0.312 | 0.0111 | 1.00 | 0.180 | 0.0302 | – | 185 | 58 |
| 2 | k3 base+cyc+force stable-only LN std-in | eqvF | mse | 6 | 0.152 ± 0.02 | 0.181 | 0.326 | 0.0110 | 1.00 | 0.192 | 0.0289 | – | 248 | 58 |
| 3 | k3 base+cyc+force stable-only LN std-in | eqvF | mse | 6 | 0.155 ± 0.02 | 0.180 | 0.334 | 0.0110 | 1.00 | 0.191 | 0.0300 | – | 215 | 63 |
