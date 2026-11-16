# Model optimisation: what changed and how much it helped

*Status: 2026-11-16. Short version; details and statistics are in `MODEL_OPTIMISATION_SUMMARY.md`.*

## What we changed, component by component

| # | Component | Before | Now | Applies to |
|---|---|---|---|---|
| 1 | **Atom description** | atom type as the number 1 or 2 | `pair_props_edge`: 11 physical properties of the atom's own element and of the partner element, the A/B role, and 2 bond features (bond length relative to the atomic radii, electronegativity difference) | both models |
| 2 | **Positional encodings** | none | none (tested twice, no reliable gain) | — |
| 3 | **Training recipe** (fixes the plateau where runs got stuck) | lr 2e-3, plateau scheduler, no clipping | lr 1e-3, gradient clipping 1.0, cosine schedule with 20 warm-up epochs (min lr 1e-5) | CGCNN (the Transformer already had this) |
| 3 | **Target scaling** | mean / std | median / MAD (robust to the few very large corrections) | CGCNN, point Transformer |
| 4 | **Architecture** | CGCNN 2 layers, raw bond length | CGCNN 4 layers, bond length expanded into 16 radial basis functions | CGCNN (Transformer size was already best) |
| 5 | **Loss weight on the total energy (λ)** | point 0.02, planar 0.01 | point 0.06, planar 0.3 | Transformer (CGCNN unchanged) |
| 6 | **Ensembling** (new option) | one model | fast mode: one model; accurate mode: mean of 5 seeds (`--job-profile global_v2_ensemble`) | all |

Not adopted after testing:
- deeper or wider Transformer;
- changing the total-energy target type, or using a squared-error loss for the total-energy term;
- the "safer" checkpoint rule;
- picking the best seed by validation error;
- averaging CGCNN and Transformer together.

## Total improvement

All four production models were retrained with every change above, on exactly the same train/validation split as the old production models. Below, old vs. new is shown on the delivery validation set, with "new" as the mean of 3 seeds.

How to read the table:
- **MAE** is an absolute error in meV/atom. Its change is given as a relative change in %.
- **R_tot** is already a percentage: the median relative error of a structure's total defect energy. "34.8 → 11.4 %" means a typical structure's total energy was off by 34.8 % before and by 11.4 % now.
- A difference between two percentages is given in **percentage points (pp)**: 34.8 % − 11.4 % = 23.4 pp. The relative change of R_tot (−67 % in this example) is shown as well.

| Model | Per-atom MAE (meV/atom) | R_tot (median total-energy error) | R_tot change |
|---|---|---|---|
| CGCNN · point | 12.0 → **5.6** (−54 %) | 34.8 % → **11.4 %** | −23.4 pp (−67 % relative) |
| CGCNN · planar | 26.5 → **6.1** (−77 %) | 32.8 % → **7.0 %** | −25.8 pp (−79 % relative) |
| Transformer · point | 7.3 → **6.9** (−5 %) | 16.0 % → **9.9 %** | −6.1 pp (−38 % relative) |
| Transformer · planar | 32.9 → **12.8** (−61 %) | 46.5 % → **7.4 %** | −39.1 pp (−84 % relative) |

- **In short:** the per-atom error is roughly halved, and the total-energy error drops by a factor of 2–6. The planar models gain the most, because they suffered most from the plateau problem.
- **The accurate (ensemble) mode improves this further.** On the independent test splits, the mean of 5 seeds beat a single model in 24 of 24 cases: R_tot −3.6 pp (percentage points) and MAE −6 % (relative) on average.
  - On the delivery validation set, compared with the average single seed:
    - planar CGCNN: R_tot 6.7 → 5.6 %, MAE 5.9 → 5.4 meV/atom;
    - planar Transformer: R_tot 7.7 → 6.6 %;
    - point models: MAE better (CGCNN 5.3 → 4.9, Transformer 7.4 → 7.0 meV/atom), but R_tot mixed. The point validation set has only 63 structures, so its median is noisy.
  - **Speed** (GPU inference mode, median per crystal, including pre- and post-processing):
    - fast mode: 2.4–3.9 ms;
    - ensemble mode: 5.8–10.8 ms, about 2–3× slower.
- **Caveat:** the delivery validation set is also used to choose the best epoch, for both old and new models, so the absolute numbers are slightly optimistic. The point set has only 63 structures. The paired test-set results from the individual studies point the same way.

## Where things are
- **Fast mode (default):** `--job-profile global_v2`. One model per job (seed 0) in the usual production checkpoint files.
- **Accurate mode:** `--job-profile global_v2_ensemble`. Mean of 5 seeds from `ensemble_checkpoints/`, written to `<job>_ensemble/` output folders.
  - Alternatively, give `--checkpoint a.pt,b.pt,...` for any ensemble.
- **Old production checkpoints:** backed up in `production_backup_2026-11-16/`. To roll back, copy them back.

Plots:
- `production_retrain/retrain_results.png`: old vs. new;
- `ensemble_study/ensemble_results.png`: ensemble size.
