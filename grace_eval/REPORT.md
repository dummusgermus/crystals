# GRACE C15 evaluation set: what we could test and how our models do

*2026-11-16. Data: `laves-ase-grace-c15/` (GRACE-2L-SMAX-large, 200 compounds). Plot: `grace_eval/grace_results.png`. Numbers: `grace_eval/results.json`.*

## What our models do (for context)
Our models take an **unrelaxed** structure together with its per-atom energies and predict how much each atom's energy changes during relaxation. The total relaxation energy is the sum of those changes.

We used the planar models, which are trained on stacking sequences in hexagonal cells. Each was run as the fast single model and as the 5-model ensemble.

## Data preparation
- **Usable compounds:** 189 of 200. The 11 sulfur compounds are excluded because S is missing from our element-property table. NdSi₂ was kept.
- **Graph format:** graphs were built with exactly the training pipeline (`planar_graph_maker`, same neighbour shells and cycle features). The totals reproduce the GRACE energies in the `.yaml` files exactly.
- **Stacking-fault labels:** the model needs to know which atoms belong to a stacking fault. I derived these from the layer stacking, with a rule that reproduces the labels of all 1978 training structures exactly. Every FFHFF cell contains exactly one fault.
- **Primitive cells:** the cubic 24-atom cell was rebuilt as the 18-atom hexagonal cell used in training. Pristine C15 has no free internal coordinates, and its per-atom energies are uniform to < 1e-6 eV, so this rebuild is exact.
- **Different interatomic potential:** the per-atom energies come from GRACE, while our training data used a different potential. The magnitudes are similar, e.g. for ZrNi₂ Zr −8.86 vs −8.72 eV and Ni −5.92 vs −6.03 eV, but they are not identical.

## Results

### 1. Stacking-fault relaxation (FFHFF, 0 % strain): this matches our task
The typical (median) GRACE relaxation energy is 0.67 meV/atom, in the same range as our training data (median 1.8 meV/atom).

| Model | Median error, total relaxation energy (meV/atom) | Better than "no relaxation" | Median relative error |
|---|---|---|---|
| "No relaxation" baseline | 0.67 | – | 100 % |
| CGCNN (ensemble) | 0.36 | 64 % of compounds | 49 % |
| **Transformer (ensemble)** | **0.12** | **88 % of compounds** | **20 %** |

- **The Transformer transfers well.** Its median error is 5× smaller than the baseline. A relative error of about 20 % lies between our own random test split (about 8 %) and our held-out-chemistry split (about 28 %).
- **The fast models are almost as good.** Ensembling helps little here; the single Transformer reaches 0.13 meV/atom and 21 %.
- **Known and new element pairs behave similarly.** The 168 pairs that occur in our training data give 0.12 meV/atom, and the 21 new pairs give 0.15 meV/atom.
- **A few outliers dominate the average error.** 10 compounds cause 94 % of the total error. SnMg₂, GeMg₂, SiMg₂, CoTi₂ and MgIn₂ have C15 structures that are far from stable in GRACE: the pristine cell alone releases up to 2 eV/atom. For these, our models predict 100–400 meV/atom of relaxation where GRACE finds only 0.5–30. The Mn compounds (Tb, Ho, Dy, Gd, Y) are the opposite case: GRACE relaxes them by 10–32 meV/atom, and we predict about 0.
- **Totals transfer, per-atom values do not.** The per-atom error (10–12 meV) is larger than that of "no relaxation" (5.8 meV). How the total energy is split into per-atom energies differs between GRACE and our training potential; that split is not a physical quantity.

### 2. Pristine C15, volume relaxation (Primitive, 0 % strain): not covered by training
- **What happens here:** the cell relaxes from the experimental lattice constant to GRACE's own (median 3.7 meV/atom, up to 2 eV/atom).
- **Result:** our models are no better than "no relaxation" (4.5 vs 3.7 meV/atom). This is expected, because our training relaxations do not include this kind of uniform volume change.

### 3. Pristine C15, already relaxed (XYZ): sanity check
- **What should happen:** the true relaxation is exactly 0.
- **Result:** the models predict about 4 meV/atom. Every training structure starts away from equilibrium, so the models always expect some relaxation. They cannot tell that a structure is already relaxed, because they never see forces.

### 4. Strain test (−10 % … +10 %, the colleague's expectation 2): fails as expected
- **The expectation:** relaxing a strained cell should give the same energy as relaxing the unstrained cell.
- **What our models do:** they predict almost no relaxation for the strained cells. The predicted relaxed energies still spread by about 785 meV/atom across strains, exactly as much as the input energies (panel C of the plot).
- **Why:** these strain energies (115–920 meV/atom) are far outside our training data, and relaxing the cell shape and volume is not part of our task.

### 5. Not testable with our models
- **Energy–strain curve (EOS):** the energy of the current configuration is an *input* to our models, not something they predict. The U-shaped curves are already present in the GRACE input energies.
- **Relaxed geometry (volume, fractional coordinates):** our models do not predict geometry.

## Why the interatomic potential matters

### What an interatomic potential is
An interatomic potential is a function that takes the atom positions and elements and returns the energy of the structure. The forces on the atoms follow from it, as the slope of that energy.
- The "true" answer would come from quantum mechanics (usually density functional theory, DFT). That is too expensive for thousands of structures with thousands of atoms.
- A potential is a cheap approximation of it.
- A relaxation means moving the atoms downhill on the potential's energy landscape until the forces are close to zero.

There are two main kinds:
- **Classical potentials** (e.g. EAM/MEAM): a fixed formula with a few parameters, fitted for each element combination to a handful of known properties.
- **Machine-learned potentials:** a flexible model trained on large amounts of DFT data.

**GRACE** ("Graph Atomic Cluster Expansion") is a machine-learned potential. GRACE-2L-SMAX-large is one of its large pretrained "universal" models, meant to cover most of the periodic table with a single model. As far as I know it was trained on large public DFT datasets; the exact training data and accuracy are not verified here.

### Why a different potential is a problem
Every potential has its own energy landscape. The same structure can get:
- a different equilibrium lattice constant;
- a different relaxation energy;
- in extreme cases, a different answer to whether the structure is stable at all.

Our model learned how atoms relax under the potential used for our training data. The reference values here are what GRACE says. Wherever the two potentials disagree, our model counts as "wrong" by GRACE's standard, even if it reproduces our own potential perfectly.

We have not yet checked which potential produced our training data. It is in the `pair_style` line of the LAMMPS inputs in `SIMULATIONS/`, which are not on the cluster. The per-atom energies are close but not identical (ZrNi₂: Zr −8.86 eV in GRACE vs −8.72 eV in our data; Ni −5.92 vs −6.03 eV), so it is almost certainly a different potential.

### How it affects the results above
1. **Totals transfer, per-atom values do not.** Only the total energy is physically meaningful. Each potential splits the total among the atoms in its own way, and that split cannot be measured. Our model learned our potential's split. This explains why the total relaxation energy transfers well (0.12 meV/atom), while the per-atom errors are worse than predicting no change.
2. **Shifted inputs.** The model receives the initial per-atom energies as input. In GRACE these are shifted by about 0.1 eV/atom compared with training. That is within the range seen in training, but the numbers differ for the same chemistry.
3. **The errors are a mixture.** Every error shown here is model error plus the disagreement between the two potentials. This data cannot separate the two, so **the numbers above are conservative**: they measure transfer to a different potential, not only the model.
4. **Some outliers are probably potential effects.**
   - SnMg₂, GeMg₂ and SiMg₂ are far from stable as C15 in GRACE: the pristine cell alone releases about 2 eV/atom.
   - For the Mn compounds, GRACE finds 10–32 meV/atom of relaxation where our model predicts almost none. Mn compounds are often magnetic, and potentials treat magnetism very differently. This is a hypothesis we have not checked.

### How to separate model error from potential differences
- **Name the gap:** find out which potential our training data used (the `pair_style` line).
- **Cleanest test:** compute some of our training structures with GRACE as well, or the colleague's structures with our potential. Disagreement between the two potentials on the same structure is not model error.
- **Longer term:** if the project moves to GRACE, retrain on GRACE-computed data. The model architecture and pipeline carry over unchanged.

## Points for the meeting
1. On the task we were trained for, the relaxation energy of stacking faults at fixed strain, the Transformer transfers to a new potential and new cells. Its median error is 0.12 meV/atom, 5× better than assuming no relaxation.
2. Strain, volume relaxation, the energy–strain curve and geometry are different tasks. Covering them would need different training data, with strained cells and cell relaxation, and probably a different model output.
3. Several "C15" compounds in this set are far from stable in GRACE, e.g. Mg₂Si-type compounds. They should probably be treated separately in any evaluation.
4. Per-atom energies are specific to each potential. Comparisons across potentials should use total energies.
5. The reference energies come from a different potential (GRACE) than our training data. The 0.12 meV/atom therefore already includes the disagreement between the two potentials. To separate model error from potential differences, the same structures should be computed with both potentials (see "Why the interatomic potential matters").

Files:
- `grace_eval/build_graphs.py`: builds the graphs.
- `grace_eval/stacking_labels.py`: computes the stacking-fault labels.
- `grace_eval/evaluate.py`: runs the models.
- `grace_eval/plot_results.py`: makes the plot.
- `grace_eval/graphs_*.pt`: the graphs.
- `grace_eval/predictions.pt`: the model predictions.
