# Model Card: AlphaFold-Bayesian

Bayesian inference of protein conformational ensembles by integrating AlphaFold-based
modeling with sparse experimental data.

**Authors:** J. Paluba and W. Ma (University of Vermont) · wen.ma@uvm.edu
**Repo:** https://github.com/Ma-Lab-biophysics/AlphaFold-Bayesian · **License:** GPL-3.0-or-later

---

## Model details

A two-stage pipeline: a pretrained deep network generates candidate conformations, and a
fitted probabilistic model (Bayesian inference) assigns them posterior weights.

**1. Sampling — AlphaFold2, used as released.** Evoformer + structure module, run through
ColabFold. Conformational diversity comes from MSA subsampling (`max-seq`,
`max-extra-seq`) and multiple random seeds, five models per seed. 
Sampling settings per system are given in Methods IV.A in the manuscript. 

**2. Reweighting — the fitted model.** Bayesian maximum-entropy reweighting: 
one free log-weight parameter per conformer (5,567 for full-length
tau; ~1.9 × 10⁴ per UvrD state), producing a normalized posterior weight for every individual structure.

---

## Training details

Pretrained AlphaFold2 is used; no fine-tuning in this work. What is fitted to data is the reweighting model of component 2. 
Bayesian inference is a machine-learning method, and that fit is described here as training in the ordinary sense:
parameters estimated from data by regularized optimization with a selected hyperparameter.

**Training data.** Two previously published datasets (external to candidate generation):

- **UvrD smFRET** (refs 22, 23) — mean efficiency and variance for the open and closed states. 
- **tau NMR-PRE** (ref 25) — amide intensity ratios at 10 MTSL sites. 

AlphaFold2's own training corpus is upstream and not audited here, but its
composition matters — see Biases.

**Objective.** Minimize `theta * S_KL + chi^2/2`, where `chi^2` is the weighted squared
deviation between ensemble-averaged and measured observables (the loss) and `S_KL` is the
Kullback–Leibler divergence from a uniform reference distribution `w0` (the regularizer).

**Procedure.** Regularized MAP estimation over log-weights, optimized by BFGS
(`scipy.optimize.fmin_bfgs`), initialized from uniform weights. All restraints are fitted
simultaneously; no held-out split is used.

**Hyperparameter.** `theta` sets the strength of the prior and is selected by scanning a
broad range.

---

## Intended use

Reweighting a pre-generated ensemble of atomistic conformations against sparse,
distance-sensitive experimental observables — demonstrated for
single-molecule FRET (mean and variance) and NMR-PRE distance restraints.

---

## Evaluation

`chi^2` (agreement with experiment) and `S_KL` (divergence from the uniform prior),
reported jointly across a `theta` scan since either alone is trivially optimizable.

Baselines: the unweighted prior ensemble; AlphaFold2's pTM ranking at default settings.

---

### Cross-validation

Restraints were grouped by NMR-PRE spin-label site and partitioned into five folds of two
sites each, so that all ten sites are held out exactly once. Each fold fits the weights on
the remaining sites at `theta = 2500` and evaluates `chi^2` on the withheld restraints.
Grouping by site rather than splitting restraints at random is necessary because restraints
sharing a spin label are strongly correlated: neighbouring residues report almost the same
probe distance, so a random split would place near-duplicate measurements in both partitions.
**The detailed results are in `examples/multi_d/split/cross_validation/`.**

Reduced `chi^2`, half-sum convention as in `summary.json`. Held-out `chi^2` exceeds the
all-data fit on the same restraints by 30.5%, the expected in-sample optimism, and is 47.4%
below the unreweighted AlphaFold prior in every fold (range 39.5-56.2%). The inferred
weights therefore predict restraints from spin labels never seen during fitting
substantially better than no reweighting.

Weight agreement with the all-data fit: on average 7.0 of the top 10 and 36.6 of the top 50
models are retained per fold, with Spearman 0.73 over the top 100. The identity of the
high-weight sub-ensemble is largely preserved; 
ensemble-level quantities are better supported than the ranking of individual models.

---

## Model examination

The model is interpretable by construction: inference returns an explicit posterior weight
for every individual conformer, so any ensemble-level property can be traced back to
identifiable structures rather than to a latent representation.

- **Weights on physical coordinates.** Projecting the posterior onto RMSD from the closed-state crystal structure and the C-alpha distance between the FRET labeling sites resolves assignable states (main-text Fig. 2A,B).
- **Agreement with data, per probe.** Posterior-weighted probe-to-residue distance profiles are compared directly against the experimental PRE distances (Fig. S4).
- **Direct inspection.** `top50_weights.dat` lists the highest-weight conformers, so users can examine the structures driving any conclusion.

---

## Biases and Limitations

- Results are bounded by sampling quality. Sampling has to be sufficient.
Verify that the candidate ensemble spans the conformational range of interest; 
the posterior redistributes weight but cannot recover an unsampled state.
- AlphaFold2's training data is dominated by well-resolved conformations — over 80% from
X-ray crystallography — biasing predictions toward crystallographically favored states and
away from flexible or disordered ones. Mitigating this is the purpose of the framework:
experimental measurements, not the network's internal confidence, set the populations. The
mitigation is bounded, since weight can only be redistributed among sampled structures.

---

## Compute infrastructure
- AlphaFold2 predictions were generated with LocalColabFold on NVIDIA V100 and 3090 GPUs. Molecular dynamics simulations were run with Amber on NVIDIA 3090 GPUs. Bayesian inference was performed on CPU.

---
