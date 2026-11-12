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

## 3. The plateau problem (in progress)

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

### Next step (running)
Before adopting the new recipe, we are checking that it doesn't hurt elsewhere. This is job 5259243, with 84 runs on at most 16 GPUs:
- **Point CGCNN:** the current recipe vs. `combo` vs. `combo-rob`.
- **Transformer, planar and point:** current vs. robust target scaling. The Transformer already uses learning rate 1e-3, clipping and a cosine schedule.

If the results hold, the new recipe becomes the CGCNN default in `delivery_global_v2.py`.

Progress: `plateau_study/progress_confirm.log`.
