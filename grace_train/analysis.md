# GRACE-only training vs. zero-shot (all 189 compounds, each tested once via 5-fold CV)

Median error of the total relaxation energy (meV/atom); R_tot over structures with |dE| > 0.5 meV/atom; 'beats 0' = share of compounds better than predicting no relaxation. Trained: checkpoint = best val R_tot, mean over 3 seeds (seed ens. = mean prediction of the 3 seeds).


## FFHFF

| model | train compounds | median err | R_tot % | beats 0 | mean err |
|---|---|---|---|---|---|
| no relaxation | – | 0.67 | 100 | – | |
| zero-shot cgcnn-ensemble5 | 0 (old data) | 0.36 | 49.3 | 0.64 | 6.8 |
| zero-shot transformer-ensemble5 | 0 (old data) | 0.12 | 20.2 | 0.88 | 7.7 |
| GRACE-trained cgcnn | 25 | 0.40 | 54.9 | 0.76 | 1.3 |
| GRACE-trained cgcnn (seed ens.) | 25 | 0.33 | 42.7 | 0.79 | 1.2 |
| GRACE-trained cgcnn | 50 | 0.33 | 40.7 | 0.75 | 1.2 |
| GRACE-trained cgcnn (seed ens.) | 50 | 0.30 | 40.3 | 0.82 | 1.1 |
| GRACE-trained cgcnn | 100 | 0.27 | 36.0 | 0.81 | 1.0 |
| GRACE-trained cgcnn (seed ens.) | 100 | 0.24 | 32.7 | 0.84 | 1.0 |
| GRACE-trained cgcnn | all | 0.25 | 32.1 | 0.79 | 1.0 |
| GRACE-trained cgcnn (seed ens.) | all | 0.22 | 25.5 | 0.84 | 0.9 |
| GRACE-trained transformer | 25 | 0.33 | 41.2 | 0.8 | 1.2 |
| GRACE-trained transformer (seed ens.) | 25 | 0.30 | 37.0 | 0.81 | 1.2 |
| GRACE-trained transformer | 50 | 0.38 | 43.6 | 0.73 | 1.2 |
| GRACE-trained transformer (seed ens.) | 50 | 0.33 | 37.0 | 0.74 | 1.2 |
| GRACE-trained transformer | 100 | 0.37 | 35.1 | 0.75 | 1.2 |
| GRACE-trained transformer (seed ens.) | 100 | 0.35 | 32.7 | 0.74 | 1.2 |
| GRACE-trained transformer | all | 0.35 | 34.6 | 0.76 | 1.2 |
| GRACE-trained transformer (seed ens.) | all | 0.33 | 30.8 | 0.76 | 1.2 |

## Primitive

| model | train compounds | median err | R_tot % | beats 0 | mean err |
|---|---|---|---|---|---|
| no relaxation | – | 3.69 | 100 | – | |
| zero-shot cgcnn-ensemble5 | 0 (old data) | 4.46 | 73.6 | 0.6 | 34.5 |
| zero-shot transformer-ensemble5 | 0 (old data) | 4.57 | 70.6 | 0.63 | 38.1 |
| GRACE-trained cgcnn | 25 | 4.28 | 98.9 | 0.57 | 41.8 |
| GRACE-trained cgcnn (seed ens.) | 25 | 3.39 | 91.2 | 0.62 | 41.3 |
| GRACE-trained cgcnn | 50 | 4.19 | 99.9 | 0.5 | 42.0 |
| GRACE-trained cgcnn (seed ens.) | 50 | 3.76 | 98.0 | 0.61 | 41.7 |
| GRACE-trained cgcnn | 100 | 4.00 | 100.0 | 0.56 | 33.9 |
| GRACE-trained cgcnn (seed ens.) | 100 | 3.67 | 99.9 | 0.62 | 33.5 |
| GRACE-trained cgcnn | all | 3.53 | 100.0 | 0.65 | 30.3 |
| GRACE-trained cgcnn (seed ens.) | all | 3.31 | 99.3 | 0.72 | 29.9 |
| GRACE-trained transformer | 25 | 3.51 | 89.4 | 0.77 | 41.4 |
| GRACE-trained transformer (seed ens.) | 25 | 3.09 | 88.0 | 0.79 | 41.1 |
| GRACE-trained transformer | 50 | 2.91 | 87.1 | 0.78 | 41.2 |
| GRACE-trained transformer (seed ens.) | 50 | 2.56 | 83.9 | 0.77 | 41.1 |
| GRACE-trained transformer | 100 | 2.88 | 84.2 | 0.81 | 41.1 |
| GRACE-trained transformer (seed ens.) | 100 | 2.62 | 83.0 | 0.81 | 41.1 |
| GRACE-trained transformer | all | 2.97 | 87.3 | 0.83 | 41.2 |
| GRACE-trained transformer (seed ens.) | all | 2.86 | 86.5 | 0.83 | 41.2 |

## XYZ

| model | train compounds | median err | R_tot % | beats 0 | mean err |
|---|---|---|---|---|---|
| no relaxation | – | 0.00 | 100 | – | |
| zero-shot cgcnn-ensemble5 | 0 (old data) | 4.22 | nan | None | 7.9 |
| zero-shot transformer-ensemble5 | 0 (old data) | 4.13 | nan | None | 8.0 |
| GRACE-trained cgcnn | 25 | 0.38 | nan |  | 2.7 |
| GRACE-trained cgcnn (seed ens.) | 25 | 1.04 | nan |  | 2.5 |
| GRACE-trained cgcnn | 50 | 0.12 | nan |  | 2.2 |
| GRACE-trained cgcnn (seed ens.) | 50 | 0.48 | nan |  | 2.2 |
| GRACE-trained cgcnn | 100 | 0.01 | nan |  | 1.3 |
| GRACE-trained cgcnn (seed ens.) | 100 | 0.09 | nan |  | 1.3 |
| GRACE-trained cgcnn | all | 0.00 | nan |  | 1.1 |
| GRACE-trained cgcnn (seed ens.) | all | 0.02 | nan |  | 1.1 |
| GRACE-trained transformer | 25 | 0.58 | nan |  | 1.0 |
| GRACE-trained transformer (seed ens.) | 25 | 0.65 | nan |  | 1.0 |
| GRACE-trained transformer | 50 | 0.69 | nan |  | 1.0 |
| GRACE-trained transformer (seed ens.) | 50 | 0.79 | nan |  | 1.0 |
| GRACE-trained transformer | 100 | 0.83 | nan |  | 1.0 |
| GRACE-trained transformer (seed ens.) | 100 | 0.86 | nan |  | 1.0 |
| GRACE-trained transformer | all | 0.66 | nan |  | 0.7 |
| GRACE-trained transformer (seed ens.) | all | 0.66 | nan |  | 0.7 |
