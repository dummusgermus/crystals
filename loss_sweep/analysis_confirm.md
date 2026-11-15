# Loss confirmation (seeds 0-8, paired vs base, checkpoint = best val R_tot)

Negative = better. p = two-sided Wilcoxon signed-rank over seeds.

| model | domain | split | variant | n | base R_tot % | var R_tot % | ΔR_tot pp | wins | p | base MAE | var MAE | ΔMAE % | p(MAE) | Δ|ΔE| % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cgcnn | point | random | flip-target | 9 | 20.4 | 15.7 | -4.66 | 7/9 | 0.0391 | 0.01 | 0.01 | +2.6 | 0.652 | -92.9 |
| cgcnn | point | chem | flip-target | 9 | 20.1 | 17.9 | -2.19 | 5/9 | 0.164 | 0.01 | 0.01 | +0.0 | 1 | -92.6 |
| cgcnn | point | **pooled** | flip-target | 18 | | | -3.43 | | 0.0139 | | | +1.3 | 0.609 | |
| transformer | point | random | lamx3 | 9 | 25.8 | 16.6 | -9.28 | 5/9 | 0.203 | 0.01 | 0.01 | -15.4 | 0.0195 | -23.4 |
| transformer | point | chem | lamx3 | 9 | 20.2 | 18.4 | -1.74 | 6/9 | 0.164 | 0.01 | 0.01 | +2.0 | 1 | +5.7 |
| transformer | point | **pooled** | lamx3 | 18 | | | -5.51 | | 0.0483 | | | -6.7 | 0.0898 | |
| transformer | planar | random | lamx10 | 9 | 36.5 | 20.6 | -15.90 | 9/9 | 0.00391 | 0.02 | 0.01 | -14.0 | 0.129 | -25.3 |
| transformer | planar | chem | lamx10 | 9 | 51.9 | 35.3 | -16.60 | 9/9 | 0.00391 | 0.03 | 0.03 | +15.4 | 0.57 | +35.0 |
| transformer | planar | **pooled** | lamx10 | 18 | | | -16.25 | | 7.63e-06 | | | +0.7 | 0.495 | |
| transformer | planar | random | lamx30 | 9 | 36.5 | 9.8 | -26.68 | 9/9 | 0.00391 | 0.02 | 0.01 | -26.7 | 0.0547 | -79.2 |
| transformer | planar | chem | lamx30 | 9 | 51.9 | 28.9 | -23.04 | 9/9 | 0.00391 | 0.03 | 0.04 | +36.6 | 0.164 | +64.3 |
| transformer | planar | **pooled** | lamx30 | 18 | | | -24.86 | | 7.63e-06 | | | +5.0 | 0.734 | |
