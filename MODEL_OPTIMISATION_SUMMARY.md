# Optimising the defect-energy GNNs: what we tested and what we kept

*Status: 2026-11-12. This document is updated step by step as we work through the pipeline components.*

## Background

The models predict a per-atom energy correction for Laves AB2 (C14/C15) defect structures. There are two datasets:
- **point**: point defects.
- **planar**: planar defects.

Two model families are used:
- **CGCNN**: a gated message-passing network.
- **Graph Transformer**: attention over all atoms, with a learned bias only on bonded (k-NN) pairs.

We judge models by two numbers:
- **R_tot**: the median relative error of the *total* defect energy, in %. This is the number we care about most.
- **MAE**: the per-atom error, in meV/atom.

Every comparison below is *paired*. A variant and the baseline get the same data split and the same random seed, so we compare like with like. We test on two kinds of split:
- **random**: random structures are held out.
- **chem**: whole element pairs are held out. This shows whether the model transfers to unseen chemistry.

---

## 1. Atom-type encoding (done)

### The problem
Each atom originally told the model only "type 1 or type 2", which is the A or B site of the AB2 lattice. That has two drawbacks:
- The model could not see *which elements* it was looking at. Ag–Be and Ba–Al looked identical.
- Writing the type as the numbers 1 and 2 suggests a fake ordering ("2 is more than 1").

### What we tested
We tried 9 ways to describe an atom:

| Encoding | What the atom "sees" |
|---|---|
| `type12` | old baseline: the number 1 or 2 |
| `none` | no type information at all (sanity check) |
| `role_onehot` | A-site or B-site as a yes/no pair (removes the fake ordering) |
| `elem_onehot` | which element it is (one slot per element, 61 elements) |
| `elem_props` | 11 physical properties of its element: atomic number, period, group, Mendeleev number, mass, electronegativity, atomic radius, molar volume, melting point, ionisation energy, electron affinity |
| `elem_props_role` | element properties + A/B role |
| `pair_props` | own element properties + those of the *partner* element in the compound + role |
| `onehot_props` | element identity + element properties + role |
| `pair_props_edge` | `pair_props`, plus two bond features: bond length relative to the sum of the atomic radii, and the electronegativity difference |

### How we tested
1. **Screening.** All 9 encodings were run on both models, both datasets and both splits. That made 336 runs of 300 epochs each, with several seeds.
2. **Confirmation.** The top 3 plus the baseline were rerun at full training length (2000 epochs). That made 96 runs.

### What worked and what didn't
- **Physical element properties clearly help.** All six encodings that include element information beat the old 1/2 label.
- **Role alone does almost nothing.** Removing the fake ordering (`role_onehot`) barely mattered.
- **Removing the type makes things worse**, as expected (`none`).
- **The gain is largest for the Transformer.** Its per-atom error dropped by about 30 %.
- **Held-out chemistry also improves.** Because the models now see physical descriptors rather than just "type 1/2", they generalise better to element pairs they have never seen.

Full-length confirmation, change vs. the old `type12` (negative = better):

| Encoding | Δ R_tot | Δ MAE | Settings improved (R_tot) |
|---|---|---|---|
| **`pair_props_edge`** | **−11.0 pp** | **−13.6 %** | **8 / 8** (p ≈ 1e-5) |
| `elem_props_role` | −9.0 pp | −17.5 % | 7 / 8 |
| `onehot_props` | −8.0 pp | −16.1 % | 6 / 8 |

### What we settled on
**`pair_props_edge` is the new default.** It describes each atom by its own element properties, the partner element's properties and its A/B role, and adds two element-aware bond features. It was the only encoding that improved R_tot in every one of the 8 settings.

The default is wired into `delivery_global_v2.py` (`ATOM_ENCODING`). It is stored in each checkpoint, so `crystal_prediction_export.py` applies the right encoding automatically. Old checkpoints still load as `type12`.

Details: `atom_encoding_sweep/ranking.md`, `atom_encoding_sweep_long/ranking_encoding.md`, and the plot `atom_encoding_sweep/best_encodings_vs_current.png`.

---

## 2. Positional encodings (done: none adopted)

### The question
At the start the models used **no positional encoding**:
- CGCNN only sees bond lengths.
- The Transformer only knows which atoms are directly bonded.

We tested whether telling the model more about *where an atom sits in the structure* helps.

### What we tested (204 runs, all with `pair_props_edge`)
- **Per-atom structural features** (both models):
  - **Random-walk features (RWSE 8 / 16):** how likely a random walk along the bonds is to return to the atom after 1…k steps. This describes the local ring/cluster structure.
  - **Laplacian eigenvectors (LapPE 8 / 16):** "vibration modes" of the bond network that give each atom smooth coordinates within the graph.
  - **The combination LapPE 8 + RWSE 16.**
- **Distance-aware attention (Transformer only):** a learned bias on *every* pair of atoms, based on one or both of:
  - the number of bonds between them (hop distance),
  - the length of the shortest bond path in Å.

  The raw simulation cells are not on the cluster and the positions are wrapped across periodic boundaries, so the bond-path length stands in for the true distance.

### Results
- **Per-atom features:** small, noisy, mixed effects. Nothing was reliably better.
  - Best: LapPE 8 + RWSE 16 on CGCNN, R_tot −4.9 pp and MAE −3 %. This is not statistically significant (p ≈ 0.055).
  - Best for the Transformer: RWSE 8, R_tot −2.6 pp but MAE +9 %.
- **Distance-aware attention: clearly worse.** R_tot was +19 to +38 pp and MAE +50 to +70 % in all settings.
  - The training curves show this is mainly a *training instability*: these models either got stuck early or blew up midway (around epoch 600).
  - So it does not prove the idea itself is useless.

### What we settled on
**No positional encoding (unchanged).**

We plan to retest LapPE 8 + RWSE 16 (CGCNN) and a stabilised version of the attention biases once training itself is stable (see section 3). Much of the noise in this sweep comes from that instability.

Details: `pe_sweep/ranking_pe.md`, plot `pe_sweep/pe_sweep_results.png`.

---

## 3. The plateau problem (done)

### What happens
Some runs, mostly CGCNN on the planar dataset, get **stuck**. For tens to hundreds of epochs the model predicts roughly the same small value for every atom. Its error sits around 34 meV/atom, compared with about 13 meV/atom for a run that trains properly. Some runs escape, some never do, and whether a run gets stuck depends on the random seed. That makes comparisons between configurations noisy.

### Why it happens (diagnosis)
Three causes reinforce each other:

1. **Very skewed targets → slow start.**
   - Planar energy corrections are mostly tiny: the median is about 4 meV.
   - A few are huge: the top 1 % exceed about 770 meV.
   - So the easiest first solution is "predict roughly the median everywhere". The model sits there for 20–150 epochs before it starts fitting the real structure.
2. **Training spikes at the current learning rate (2e-3).**
   - Occasional large gradient jumps knock a model that was already learning back onto the plateau.
   - In one diagnostic run a single spike at epoch 60 left the model permanently stuck.
3. **The scheduler and the checkpoint choice make it worse.**
   - The learning-rate scheduler sees "no improvement" during the plateau and shrinks the learning rate, so escaping becomes even harder.
   - We keep the epoch with the best *validation R_tot*. On the plateau, R_tot can look deceptively good by chance, so we sometimes keep an epoch from the stuck phase even though the same run later got much better.
   - Example: a checkpoint from epoch 36 with 32 meV/atom was kept, although the run later reached 13 meV/atom.

### What we are testing right now
This is a study of 132 runs on planar CGCNN: 11 variants × 6 seeds × 2 splits, 2000 epochs each, using at most 16 GPUs. Each variant targets one of the causes above:

| Variant | Idea | Targets cause |
|---|---|---|
| `base` | current recipe (reference) | — |
| `rob` | scale targets by median/MAD instead of mean/std (robust to the heavy tail) | 1 |
| `msewu` | squared-error loss for the first 30 epochs (pushes the model off the median faster), then the normal loss | 1 |
| `bn` | batch normalisation in the network | 1, 2 |
| `lr1e-3`, `lr5e-4` | lower learning rate | 2 |
| `clip` | gradient clipping (caps the size of any single update) | 2 |
| `pat30` | far more patient learning-rate scheduler | 3 |
| `cos` | fixed cosine learning-rate schedule with warm-up, independent of validation | 2, 3 |
| `combo` | lower LR + clipping + cosine schedule | 2, 3 |
| `combo-rob` | `combo` + robust target scaling | 1, 2, 3 |

**A safer checkpoint rule is tested on every run as well.** Pick the best-R_tot epoch only among epochs whose per-atom error is within 20 % of the run's best. This rules out stuck epochs.

### Results (132 runs, planar CGCNN)

The clear winner is **`combo-rob`**: learning rate 1e-3, gradient clipping, a cosine schedule with warm-up, and median/MAD target scaling. Compared with the current recipe:
- **R_tot −23 pp** (49.8 % → 26.9 %) and **per-atom MAE −51 %** (22.9 → 11.8 meV/atom).
- It was better in all 12 paired runs (p ≈ 2e-4).
- On the random split the MAE drops from 14.4 to 6.1 meV/atom.

Plain `combo` (the same without robust scaling) is almost as good: −17 pp and −51 %.

**Why it works.** The current recipe effectively stops learning early. The scheduler halves the learning rate during the slow start, so the best checkpoint lands between epochs 1 and 180. With the new recipe, runs keep improving until epochs 700–1600.

**What did not work on its own:**
- Lower learning rate and gradient clipping each cut the MAE by 20–34 % but barely moved R_tot.
- The safer checkpoint rule alone cut the MAE by 12 % and left R_tot unchanged. With the new recipe it no longer matters, because training is stable.
- Cosine schedule at the old learning rate, the patient scheduler, MSE warm-up and batch norm all made things worse.

The fixes only work together.

Details: `plateau_study/analysis.md`, plot `plateau_study/plateau_results.png`.

### Confirmation and what we settled on (84 runs)
- **Point CGCNN:** the new recipe also helps, with R_tot −4.4 pp (28.1 → 23.7 %) and MAE −12 % (p ≈ 0.01).
- **Transformer:** it already used learning rate 1e-3, clipping and a cosine schedule, which is why it never had the plateau problem. So only robust scaling was tested:
  - point: R_tot −12 pp (32.4 → 20.4 %, p ≈ 0.003), MAE unchanged;
  - planar: mixed (MAE −16 %, R_tot not better).

**New defaults** in `delivery_global_v2.py`:
- **CGCNN:** learning rate 1e-3, gradient clipping 1.0, cosine schedule (20 warm-up epochs, minimum learning rate 1e-5).
- **Target scaling:** median/MAD for CGCNN (both datasets) and the point Transformer. The planar Transformer keeps mean/std. The choice is stored in every checkpoint, and the export picks it up automatically.
- **Checkpoint rule:** unchanged (best validation R_tot). With stable training the safer rule brought nothing and hurt the planar Transformer.

Production checkpoints have not been retrained yet.

Details: `plateau_study/analysis_confirm.md`.

---

## 4. Architecture and bond-distance features (done)

Job 5264402 runs 216 runs, all with the new recipe, 3 seeds × both datasets × both splits:
- **CGCNN:**
  - depth: 2 (current), 3, 4 or 6 layers;
  - width: 64, 128 (current) or 256;
  - bond length either raw (current) or expanded into 16 / 32 Gaussian radial basis functions;
  - two combinations: 4 layers at width 256, and 4 layers with radial basis 16.
- **Transformer:**
  - depth: 2, 4 (current) or 6 layers;
  - width: 64, 128 (current) or 256;
  - radial basis 16 / 32;
  - one combination: 6 layers with radial basis 16.

### Results (216 runs)
- **CGCNN: deeper is better.** With 2 layers each atom only sees atoms two bonds away.
  - 4 layers: R_tot −4.1 pp, better in all 4 settings (p ≈ 5e-4).
  - 4 layers + radial basis 16: R_tot −4.1 pp and MAE −11.5 %.
  - 3 layers: −3.2 pp.
  - Width hardly matters, though 64 is worse.
  - Radial-basis distances alone give little; they help in combination with depth.
- **Transformer:** the current size (4 layers, width 128, raw distance) is already the best. Every change made R_tot worse.

### Confirmation and what we settled on (9 seeds, 144 runs)
**New CGCNN default: 4 layers plus bond distances as 16 radial basis functions.**
- **Against the old 2-layer model:** R_tot −4.2 pp (better in 4/4 settings, p ≈ 2e-5) and MAE −11.5 % (p ≈ 7e-5).
- **Against plain 4 layers:** the same R_tot, but MAE −9 % in all 4 settings.
- **Example:** on the planar random split, R_tot went from 17.8 to 12.0 % and MAE from 6.3 to 5.3 meV/atom.
- **Transformer:** stays unchanged.

The setting is stored in every checkpoint, and the export applies it automatically.

Details: `arch_sweep/ranking_variant.md`, `arch_sweep/confirm/ranking_variant.md`, plot `arch_sweep/arch_results.png`.

---

## 5. Positional encodings, rechecked (done: none adopted)
We compared Laplacian 8 + random-walk 16 with no positional encoding on the new CGCNN default, with 6 seeds × 4 settings (48 runs).
- **Result:** R_tot −1.0 pp (p ≈ 0.10) and MAE +4.8 %.
- **Why the earlier hint disappeared:** on the old 2-layer model this combination gave −4.9 pp. The deeper model already picks up that structure on its own.
- **Default:** stays without positional encoding.

Details: `pe_recheck/ranking_pe.md`.

---

## 6. Loss function (done: Transformer weights raised)
The loss is the per-atom error plus λ × the total-energy error, balanced so that λ is the relative weight of the total-energy term. Before this step λ was only 0.005–0.02, depending on model and dataset.

### What we tested (216 runs, 3 seeds × 4 settings, both models)
- λ at ×0 (no total-energy term), ×0.3, ×3, ×10, ×30 and ×100 of the old value;
- the total-energy target flipped (graph sum ↔ full cell);
- a squared-error instead of a Huber total-energy loss.

### Results
- More weight on the total energy lowers R_tot, but for CGCNN it raises the per-atom MAE. CGCNN had nothing that was clearly better overall.
- Removing the total-energy term (λ = 0) makes the Transformer worse, so the term is useful.
- The squared-error loss made no real difference.
- The best candidates went into a confirmation with 6 more seeds (9 in total, 84 extra runs): `loss_sweep/analysis_confirm.md`.

### Confirmation and what we settled on (9 seeds)
| Model / dataset | Candidate | R_tot random | R_tot held-out | MAE | Decision |
|---|---|---|---|---|---|
| Transformer point | λ ×3 (0.02 → 0.06) | −9.3 pp | −1.7 pp | −6.7 % (pooled) | **adopted** |
| Transformer planar | λ ×10 (→ 0.1) | −15.9 pp (9/9 seeds) | −16.6 pp (9/9) | ≈ 0 | |
| Transformer planar | λ ×30 (0.01 → 0.3) | 36.5 → 9.8 % (9/9 seeds) | 51.9 → 28.9 % (9/9) | random −27 %, held-out median unchanged | **adopted** |
| CGCNN point | graph-sum target | −0.9 pp* | −1.3 pp* | ≈ 0 | not adopted |

\*The first analysis showed a −4.7 pp gain for the graph-sum target. That turned out to be a measurement artefact: the variant was scored with the graph-sum metric and the baseline with the full-cell metric. Scored the same way, the change is neutral (and slightly worse with the full-cell metric), so CGCNN point keeps the full-cell target.

The held-out MAE of the planar Transformer jumps between seeds by a factor of 7 (9–96 meV), with or without the change. The mean MAE moves with single bad seeds, so we judged it by the median, which is unchanged. This seed-to-seed instability of the Transformer is a good reason to look at ensembling next.

New defaults in `delivery_global_v2.py`: `LAMBDA_TRANSFORMER_BY_DOMAIN = {point: 0.06, planar: 0.3}`. CGCNN is unchanged.

## 7. Production retrain (done: deployed 2026-11-16)
All four production models were retrained with every new default: the atom encoding, the plateau fix, the RBF bond features and 4 CGCNN layers, the robust target scaling and the new loss weights.
- Each was trained with 3 seeds on exactly the same train/validation split as the current production checkpoint.
- Both old and new were scored on the delivery validation set (`production_retrain/comparison.md`, plot `production_retrain/retrain_results.png`).

| Model | val MAE old → new (meV/atom) | R_tot graph-sum old → new | R_tot full-cell old → new |
|---|---|---|---|
| CGCNN point | 12.0 → 5.6 (−54 %) | 30.3 → 12.1 % | 34.8 → 11.4 % |
| CGCNN planar | 26.5 → 6.1 (−77 %) | 32.8 → 7.0 % | same |
| Transformer point | 7.3 → 6.9 (−5 %) | 16.0 → 9.9 % | 21.3 → 18.2 % |
| Transformer planar | 32.9 → 12.8 (−61 %) | 46.5 → 7.4 % | same |

Every model improves on every main metric, and the planar models improve by 4–6×. The one exception is the mean absolute total-energy error of the Transformer point model, which rises 21 %; its median relative error falls.

Caveats:
- The point validation set is small (63 structures).
- The validation set is also used to pick the best epoch, for both old and new, so these numbers are slightly optimistic. The independent test numbers from the sweeps above are the cleaner measure.

The new checkpoints are in `production_retrain/ckpts/`. **Deployed on 2026-11-16:** seed 0 replaced the four production checkpoint files (fast mode), and the old files are backed up in `production_backup_2026-11-16/`.

## 8. Ensembling (done)
### The question
Does averaging the predictions of several models help? The models are identical except for their random starting point. In particular, does it help the Transformer, whose results change a lot from seed to seed?

### How we tested (120 runs)
For each model, dataset and split type, we trained 5 models on exactly the same split with different seeds, and repeated this for 3 different splits. All runs use the current defaults. We then combined their test predictions after training (`ensemble_study/analyze.py`, results in `ensemble_study/analysis.md`).

### Results (`ensemble_study/ensemble_results.png`)
- **Averaging works everywhere.** Averaging 5 models beats the average single model in all 24 of 24 splits, on both R_tot (−3.6 pp on average, range −1.8 to −5.1 pp) and per-atom MAE (−6 %, range −2 to −10 %).
- **Most of the gain comes from the first few models.** 3 models give about 75 % of the gain of 5 (R_tot −2.7 pp, MAE −4.9 %); the curve is still slowly improving at 5.
- **Picking the best seed by validation error does not work.** That is what we did so far. It gives no reliable gain (R_tot −0.5 pp, n.s.) and on average a *worse* MAE (+4 %). For the planar Transformer on held-out chemistry, the MAE gets 31 % worse. The validation set is too small to identify a genuinely better seed.
- **Variations on the plain mean did not add anything.** We tried the median instead of the mean, averaging the checkpoints picked by best MAE instead of best R_tot, and averaging both checkpoint types. None was reliably better, and the median was slightly worse on R_tot.
- **Averaging CGCNN and Transformer** gave no reliable gain over the better single-architecture ensemble (6 of 12 splits better).

### What we settled on
Use a mean ensemble of several seeds (3–5) of the same model instead of a single "best" seed.

On the delivery split, the 3 retrained seeds per production model already form a 3-member ensemble (`production_retrain/ensemble_eval.md`). Compared with the average single retrained model:
- planar CGCNN: R_tot 7.0 → 5.9 %, MAE 6.1 → 5.7 meV;
- planar Transformer: R_tot 7.4 → 6.5 %;
- point CGCNN: MAE 5.6 → 5.2 meV;
- point Transformer: MAE 6.9 → 6.6 meV, but R_tot 9.9 → 13.2 %. This validation set has only 63 structures, so the median is noisy there; the 24-split test result is the more reliable evidence.

**Deployed as an optional accurate mode (2026-11-16).** Two more seeds were trained, so each model has 5 seeds in `ensemble_checkpoints/`. Use `crystal_prediction_export.py --job-profile global_v2_ensemble`, or give comma-separated `--checkpoint` paths.
- The export averages the members (`EnsembleModel`). This was checked against the mean of the individual models (differences of about 1e-7 eV).
- **Cost:** model time grows linearly with the number of members. The whole GPU inference per crystal goes from 2.4–3.9 ms to 5.8–10.8 ms.
- Full numbers: `production_retrain/ensemble_eval.md`.
