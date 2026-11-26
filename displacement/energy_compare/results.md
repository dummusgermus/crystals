# Relaxation energy: deployed CGCNN / Graph Transformer vs unified PaiNN

Same test structures (our 3 union compound splits, test compounds unseen), same full-cell labels (point: exact LAMMPS ΔE, planar: cell ΔE), stable structures. CGCNN / GT: deployed recipe (delivery_global_v2, residual per-atom target, 2000 epochs) retrained on our train/val compounds. PaiNN: unified production config from runs_unified (λ 0.3 + harmonic residual, mixed-trained).

Test structures per split: sp0 point 82 / planar 270, sp1 point 91 / planar 299, sp2 point 80 / planar 270

| model | point: median rel. error | point: MAE (meV) | planar: median rel. error | planar: MAE (meV) |
|---|---|---|---|---|
| **single model (3 splits × 2 seeds)** | | | | |
| CGCNN | 17.7% ± 3.1 | 415 ± 407 | 26.4% ± 4.3 | 140 ± 237 |
| Graph Transformer | 20.7% ± 3.8 | 219 ± 170 | 29.7% ± 8.1 | 35 ± 9 |
| unified PaiNN | 6.9% ± 0.5 | 76 ± 45 | 9.7% ± 2.3 | 27 ± 16 |
| **2-seed ensemble (mean over 3 splits)** | | | | |
| CGCNN | 14.8% ± 1.6 | 401 ± 240 | 24.2% ± 1.9 | 137 ± 153 |
| Graph Transformer | 17.7% ± 3.2 | 207 ± 136 | 26.8% ± 6.3 | 33 ± 7 |
| unified PaiNN | 6.9% ± 0.3 | 73 ± 38 | 9.0% ± 2.0 | 27 ± 16 |
