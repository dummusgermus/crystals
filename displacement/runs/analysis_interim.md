# Displacement study: interim

r_disp = per-structure |error| / |true displacement| (median; 1 = predicting no motion, 0 = perfect); skill = 1 - SSE/SS (pooled); mae in A. Stable moving structures unless '_all'. Mean over seeds; ranked by validation r_disp. Test split = unseen compounds.


**point reference 'd = c * F0'** (c = 0.1553 A^2/eV): val r_disp 0.634, test r_disp 0.623, test skill 0.480, test mae 0.0126 (zero-motion mae 0.0154)


## point · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rt6.4/buf4 k3 base+cyc+force | eqv | mse | 2 | 0.121 ± 0.00 | 0.195 | 0.788 | 0.0090 | 1.00 | 0.244 | – | 0.1098 | 290 | 22 |
| 2 | rt6.4/buf4 k3 base+cyc | eqv | mse | 2 | 0.121 ± 0.00 | 0.221 | 0.720 | 0.0112 | 1.00 | 0.270 | – | 0.2846 | 285 | 22 |
| 3 | rt8.9/buf4 k3 base+cyc+force | eqvF+F | mse | 1 | 0.136 ± – | 0.196 | 0.738 | 0.0079 | 0.99 | 0.259 | – | 0.2302 | 255 | 38 |
| 4 | rt8.9/buf0 k3 base+cyc+force | eqv | mse | 2 | 0.143 ± 0.00 | 0.244 | 0.753 | 0.0081 | 0.99 | 0.285 | – | 0.4980 | 265 | 21 |
| 5 | rt8.9/buf4 k2 base+cyc+force | eqv | mse | 1 | 0.152 ± – | 0.262 | 0.755 | 0.0077 | 0.99 | 0.342 | – | 0.3864 | 280 | 33 |
| 6 | rt8.9/buf4 k3 base+cyc+force+dvec | eqv | mse | 2 | 0.159 ± 0.01 | 0.239 | 0.764 | 0.0074 | 0.99 | 0.278 | – | 0.2834 | 322 | 31 |
| 7 | rt8.9/buf4 k3 base+cyc+force | eqv | mse | 1 | 0.159 ± – | 0.281 | 0.782 | 0.0066 | 0.99 | 0.352 | – | 0.1000 | 265 | 35 |
| 8 | rt8.9/buf4 k3 base+cyc+force | eqv+F | mse | 1 | 0.161 ± – | 0.222 | 0.766 | 0.0084 | 0.99 | 0.265 | – | 0.1856 | 265 | 38 |
| 9 | rt8.9/buf4 k2 base+cyc | eqv | mse | 2 | 0.164 ± 0.01 | 0.299 | 0.696 | 0.0097 | 0.99 | 0.352 | – | 0.4256 | 320 | 28 |
| 10 | rt8.9/buf4 k3 base+cyc+dvec | eqv | mse | 1 | 0.172 ± – | 0.261 | 0.763 | 0.0098 | 0.99 | 0.386 | – | 0.9498 | 310 | 35 |
| 11 | rt8.9/buf4 k8 base+cyc+force | eqv | mse | 1 | 0.176 ± – | 0.254 | 0.775 | 0.0075 | 0.99 | 0.285 | – | 0.7987 | 275 | 44 |
| 12 | rt8.9/buf0 k3 base+cyc | eqv | mse | 2 | 0.177 ± 0.02 | 0.239 | 0.778 | 0.0065 | 0.99 | 0.286 | – | 0.1181 | 285 | 21 |
| 13 | rt8.9/buf4 k1 base+cyc | eqv | mse | 2 | 0.178 ± 0.01 | 0.290 | 0.730 | 0.0106 | 0.99 | 0.321 | – | 0.7245 | 248 | 25 |
| 14 | rt8.9/buf4 k3 base+cyc | eqv | mse | 1 | 0.179 ± – | 0.277 | 0.778 | 0.0066 | 0.99 | 0.329 | – | 0.0741 | 245 | 35 |
| 15 | rt8.9/buf4 k3 base | eqv | mse | 1 | 0.186 ± – | 0.259 | 0.753 | 0.0072 | 0.99 | 0.299 | – | 0.2305 | 325 | 35 |
| 16 | rt8.9/buf4 k3 base+cyc+dirs | direct | mse | 1 | 0.198 ± – | 0.295 | 0.327 | 0.0120 | 0.97 | 0.341 | – | 0.6856 | 230 | 37 |
| 17 | rt8.9/buf4 k1 base+cyc+force | eqv | mse | 2 | 0.216 ± 0.01 | 0.285 | 0.709 | 0.0132 | 0.98 | 0.330 | – | 1.6965 | 285 | 25 |

**planar reference 'd = c * F0'** (c = 0.0773 A^2/eV): val r_disp 0.684, test r_disp 0.715, test skill 0.260, test mae 0.0155 (zero-motion mae 0.0190)


## planar · cgcnn

| # | view | head | loss | n | val r_disp | test r_disp | test skill | test mae | test cos | test r_disp (all) | cell mae | null max | ep | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3 base+cyc+force | eqvF+F | mse | 1 | 0.183 ± – | 0.213 | 0.582 | 0.0078 | 1.00 | 0.213 | 0.0341 | – | 190 | 21 |
| 2 | k3 base+cyc+force | eqv+F | mse | 1 | 0.188 ± – | 0.198 | 0.524 | 0.0082 | 1.00 | 0.198 | 0.0527 | – | 165 | 21 |
| 3 | k3 base+cyc+dirs | direct | mse | 1 | 0.216 ± – | 0.191 | 0.182 | 0.0109 | 1.00 | 0.191 | 0.0439 | – | 180 | 21 |
| 4 | k2 base+cyc+force | eqv | mse | 1 | 0.245 ± – | 0.265 | 0.543 | 0.0083 | 1.00 | 0.265 | 0.0373 | – | 220 | 23 |
| 5 | k3 base+cyc+force+dvec | eqv | mse | 2 | 0.266 ± 0.00 | 0.265 | 0.559 | 0.0086 | 1.00 | 0.265 | 0.0575 | – | 188 | 18 |
| 6 | k5 base+cyc+force | eqv | mse | 1 | 0.266 ± – | 0.251 | 0.558 | 0.0085 | 1.00 | 0.251 | 0.0367 | – | 200 | 23 |
| 7 | k3 base+cyc+force stable-only | eqv | mse | 1 | 0.276 ± – | 0.265 | 0.541 | 0.0091 | 1.00 | 0.265 | 0.0618 | – | 180 | 19 |
| 8 | k2 base+cyc | eqv | mse | 2 | 0.277 ± 0.02 | 0.273 | 0.497 | 0.0089 | 1.00 | 0.273 | 0.0469 | – | 170 | 18 |
| 9 | k8 base+cyc+force | eqv | mse | 1 | 0.277 ± – | 0.257 | 0.534 | 0.0088 | 1.00 | 0.257 | 0.0307 | – | 225 | 21 |
| 10 | k3 base | eqv | mse | 2 | 0.278 ± 0.01 | 0.296 | 0.528 | 0.0093 | 1.00 | 0.296 | 0.0653 | – | 182 | 27 |
| 11 | k5 base+cyc | eqv | mse | 1 | 0.300 ± – | 0.310 | 0.511 | 0.0094 | 1.00 | 0.310 | 0.0247 | – | 110 | 23 |
| 12 | k3 base+cyc+dvec | eqv | mse | 2 | 0.303 ± 0.01 | 0.298 | -0.241 | 0.0099 | 1.00 | 0.298 | 0.0728 | – | 222 | 27 |
| 13 | k3 base+cyc+force | eqv | mse | 2 | 0.304 ± 0.01 | 0.302 | 0.537 | 0.0093 | 1.00 | 0.302 | 0.0497 | – | 102 | 27 |
| 14 | k3 base+cyc stable-only | eqv | mse | 1 | 0.305 ± – | 0.360 | 0.525 | 0.0097 | 1.00 | 0.360 | 0.0450 | – | 105 | 19 |
| 15 | k8 base+cyc | eqv | mse | 1 | 0.307 ± – | 0.349 | 0.509 | 0.0093 | 1.00 | 0.349 | 0.0233 | – | 125 | 23 |
| 16 | k1 base+cyc+force | eqv | mse | 2 | 0.307 ± 0.01 | 0.270 | 0.523 | 0.0089 | 1.00 | 0.270 | 0.0454 | – | 155 | 18 |
| 17 | k3 base+cyc | eqv | mse | 2 | 0.310 ± 0.00 | 0.286 | 0.445 | 0.0093 | 1.00 | 0.286 | 0.0484 | – | 202 | 27 |
| 18 | k3 base+cyc | eqv | l1 | 1 | 0.318 ± – | 0.323 | 0.516 | 0.0098 | 1.00 | 0.323 | 0.0399 | – | 50 | 19 |
| 19 | k1 base+cyc | eqv | mse | 2 | 0.327 ± 0.00 | 0.336 | 0.494 | 0.0096 | 1.00 | 0.336 | 0.0327 | – | 262 | 18 |
| 20 | k3 base+cyc | eqv | huber | 1 | 0.334 ± – | 0.349 | 0.538 | 0.0094 | 1.00 | 0.349 | 0.0295 | – | 90 | 19 |
