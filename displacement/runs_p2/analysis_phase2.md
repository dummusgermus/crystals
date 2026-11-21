# Displacement study: phase2

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k3 base+cyc+force stable-only LN std-in | eqv+F | mse | 6 | 0.118 ± 0.02 | 0.209 | 0.744 | 0.0071 | 1.00 | 0.302 | – | 0.0116 | 269 | 27 |
| 2 | rt8.9/buf4 k3 base+cyc+force stable-only std-in | eqv+F | mse | 6 | 0.118 ± 0.01 | 0.217 | -197.655 | 0.0801 | 1.00 | 0.333 | – | 2.1197 | 294 | 27 |
| 3 | rt8.9/buf4 k3 base+cyc+force stable-only | eqv+F | mse | 6 | 0.123 ± 0.02 | 0.202 | -119548.407 | 1.4365 | 1.00 | 0.320 | – | 57.9778 | 327 | 57 |
| 4 | rt8.9/buf4 k2 base+cyc+force LN std-in | eqv+F | mse | 6 | 0.125 ± 0.01 | 0.222 | 0.709 | 0.0086 | 1.00 | 0.332 | – | 0.0602 | 285 | 24 |
| 5 | rt8.9/buf4 k3 base+cyc+force LN std-in | eqv+F | mse | 6 | 0.131 ± 0.01 | 0.213 | 0.580 | 0.0090 | 0.99 | 0.346 | – | 0.0292 | 264 | 28 |
| 6 | rt8.9/buf4 k3 base+force | eqv+F | mse | 6 | 0.136 ± 0.01 | 0.239 | -79.741 | 0.0797 | 0.99 | 0.349 | – | 16.4771 | 306 | 63 |
| 7 | rt8.9/buf4 k2 base+cyc+force | eqv+F | mse | 6 | 0.137 ± 0.01 | 0.234 | -0.236 | 0.0148 | 0.99 | 0.359 | – | 0.3921 | 277 | 46 |
| 8 | rt8.9/buf4 k3 base+cyc+force stable-only LN | eqv+F | mse | 6 | 0.139 ± 0.01 | 0.235 | 0.722 | 0.0086 | 0.99 | 0.341 | – | 0.0305 | 294 | 27 |
| 9 | rt8.9/buf4 k3 base+cyc+force | eqv+F | mse | 6 | 0.141 ± 0.01 | 0.233 | -30178.666 | 0.9746 | 0.99 | 0.365 | – | 64.0912 | 331 | 47 |
| 10 | rt8.9/buf4 k3 base+cyc+force | eqvF+F | mse | 6 | 0.142 ± 0.01 | 0.240 | -21752.165 | 0.5982 | 0.99 | 0.361 | – | 59.0356 | 302 | 70 |
| 11 | rt8.9/buf6 k3 base+cyc+force | eqv+F | mse | 6 | 0.149 ± 0.01 | 0.240 | -353.632 | 0.0861 | 0.99 | 0.351 | – | 3.9974 | 276 | 63 |
| 12 | rt8.9/buf4 k3 base+cyc | eqv | mse | 6 | 0.163 ± 0.01 | 0.257 | -422.217 | 0.0856 | 0.99 | 0.411 | – | 15.4125 | 270 | 70 |
| 13 | rt6.4/buf4 k3 base+cyc+force stable-only | eqv+F | mse | 6 | 0.321 ± 0.01 | 0.348 | -27568.685 | 0.6024 | 0.27 | 0.411 | – | 33.0990 | 239 | 40 |
| 14 | rt6.4/buf4 k3 base+cyc+force | eqv+F | mse | 6 | 0.323 ± 0.01 | 0.357 | -1.671 | 0.0167 | 0.24 | 0.426 | – | 0.6235 | 251 | 45 |
| 15 | rt8.9/buf4 k3 base+cyc+force [h256l6] | eqv+F | mse | 6 | 0.416 ± 0.01 | 0.468 | 0.534 | 0.0124 | 0.91 | 0.558 | – | 0.0242 | 320 | 93 |

## point · transformer

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt8.9/buf4 k2 base+cyc+force | eqv+F | mse | 6 | 0.219 ± 0.03 | 0.283 | -7.284 | 0.0310 | 0.98 | 0.378 | – | 1.0247 | 266 | 57 |
| 2 | rt8.9/buf4 k3 base+cyc+force | eqv+F | mse | 6 | 0.301 ± 0.03 | 0.355 | -0.644 | 0.0172 | 0.97 | 0.438 | – | 0.3758 | 182 | 57 |
| 3 | rt8.9/buf4 k2 base+cyc+force std-in | eqv+F | mse | 6 | 0.383 ± 0.02 | 0.444 | 0.225 | 0.0129 | 0.93 | 0.526 | – | 0.0354 | 150 | 16 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force stable-only | eqv+F | mse | 6 | 0.177 ± 0.02 | 0.202 | 0.364 | 0.0116 | 1.00 | 0.220 | 0.0395 | – | 178 | 38 |
| 2 | k3 base+cyc+force [wd1e-3] | eqv+F | mse | 6 | 0.178 ± 0.02 | 0.193 | 0.303 | 0.0122 | 1.00 | 0.208 | 0.0807 | – | 198 | 32 |
| 3 | k3 base+cyc+force std-in | eqv+F | mse | 6 | 0.185 ± 0.01 | 0.214 | 0.355 | 0.0118 | 1.00 | 0.232 | 0.0433 | – | 192 | 35 |
| 4 | k3 base+cyc+force | eqv+F | mse | 6 | 0.186 ± 0.02 | 0.219 | 0.256 | 0.0130 | 1.00 | 0.236 | 0.1103 | – | 172 | 55 |
| 5 | k3 base+cyc+force head-edges k5 | eqv+F | mse | 6 | 0.186 ± 0.02 | 0.209 | -95.182 | 0.0183 | 1.00 | 0.222 | 0.1109 | – | 221 | 38 |
| 6 | k3 base+cyc+force LN | eqv+F | mse | 6 | 0.188 ± 0.01 | 0.218 | 0.259 | 0.0132 | 1.00 | 0.235 | 0.0737 | – | 203 | 35 |
| 7 | k3 base+force | eqv+F | mse | 6 | 0.188 ± 0.02 | 0.219 | 0.291 | 0.0124 | 1.00 | 0.236 | 0.0840 | – | 196 | 35 |
| 8 | k2 base+cyc+force | eqv+F | mse | 6 | 0.189 ± 0.02 | 0.220 | 0.276 | 0.0126 | 1.00 | 0.233 | 0.1083 | – | 164 | 32 |
| 9 | k3 base+force LN std-in | eqv+F | mse | 6 | 0.193 ± 0.01 | 0.216 | 0.347 | 0.0121 | 1.00 | 0.228 | 0.0506 | – | 173 | 26 |
| 10 | k3 base+cyc+force LN std-in | eqv+F | mse | 6 | 0.195 ± 0.01 | 0.214 | 0.346 | 0.0121 | 1.00 | 0.228 | 0.0368 | – | 188 | 36 |
| 11 | k3 base+cyc+force | eqvF+F | mse | 6 | 0.197 ± 0.02 | 0.209 | 0.310 | 0.0124 | 1.00 | 0.220 | 0.0765 | – | 162 | 32 |
| 12 | k3 base+cyc+force stable-only LN std-in | eqv+F | mse | 6 | 0.198 ± 0.01 | 0.231 | 0.369 | 0.0115 | 1.00 | 0.240 | 0.0313 | – | 180 | 26 |
| 13 | k3 base+cyc | eqv | mse | 6 | 0.286 ± 0.01 | 0.314 | 0.216 | 0.0137 | 1.00 | 0.331 | 0.0771 | – | 198 | 33 |
| 14 | k3 base+cyc+force [l6] | eqv+F | mse | 6 | 0.323 ± 0.09 | 0.367 | 0.274 | 0.0141 | 1.00 | 0.383 | 0.0352 | – | 218 | 41 |
| 15 | k3 base+cyc+force [h256l6] | eqv+F | mse | 6 | 0.458 ± 0.18 | 0.472 | 0.249 | 0.0152 | 0.99 | 0.491 | 0.0354 | – | 249 | 44 |

## planar · transformer

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force | eqv+F | mse | 6 | 0.265 ± 0.04 | 0.291 | 0.315 | 0.0131 | 1.00 | 0.306 | 0.0569 | – | 298 | 47 |
| 2 | k3 base+cyc+force std-in | eqv+F | mse | 6 | 0.292 ± 0.02 | 0.334 | 0.324 | 0.0135 | 1.00 | 0.349 | 0.0475 | – | 227 | 33 |
