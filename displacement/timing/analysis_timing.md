# Displacement study: timing

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force | eqv | mse | 1 | 0.485 ± – | 0.492 | 0.629 | 0.0119 | 0.95 | 0.527 | – | 0.0703 | 10 | 2 |
| 2 | rt11/buf3 k8 base+cyc+force | eqv | mse | 1 | 1.400 ± – | 1.748 | -5.153 | 0.0541 | -0.37 | 1.576 | – | 0.4113 | 10 | 3 |

## point · transformer

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force | eqv | mse | 1 | 0.541 ± – | 0.595 | 0.566 | 0.0131 | 0.95 | 0.610 | – | 0.0623 | 10 | 2 |
| 2 | rt11/buf3 k8 base+cyc+force | eqv | mse | 1 | 0.801 ± – | 0.851 | 0.253 | 0.0115 | 0.77 | 0.864 | – | 0.0335 | 5 | 2 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force | eqv | mse | 1 | 0.378 ± – | 0.366 | 0.528 | 0.0096 | 1.00 | 0.366 | 0.0280 | – | 15 | 2 |
| 2 | k8 base+cyc+force | eqv | mse | 1 | 0.627 ± – | 0.502 | 0.463 | 0.0110 | 1.00 | 0.502 | 0.0192 | – | 20 | 2 |

## planar · transformer

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force | eqv | mse | 1 | 0.531 ± – | 0.550 | 0.380 | 0.0127 | 1.00 | 0.550 | 0.0241 | – | 20 | 3 |
| 2 | k8 base+cyc+force | eqv | mse | 1 | 0.534 ± – | 0.461 | 0.445 | 0.0113 | 1.00 | 0.461 | 0.0308 | – | 20 | 3 |
