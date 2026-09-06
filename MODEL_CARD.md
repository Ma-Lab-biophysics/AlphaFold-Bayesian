# Model Card: AlphaFold-Bayesian

Bayesian inference of protein conformational ensembles by integrating AlphaFold-based
modeling with sparse experimental data.

**Authors:** J. Paluba, C. Berger, W. Ma (University of Vermont) · wen.ma@uvm.edu
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


