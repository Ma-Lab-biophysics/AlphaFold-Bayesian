# Cross-validation — holding out experimental data

Supplementary analyses of the tau NMR-PRE Bayesian reweighting.  
These characterise how the inference behaves when data components are withheld.
This analysis uses `theta = 2500` and the full 5,567-model tau ensemble.

**Design.** Restraints are grouped by NMR-PRE spin-label site and partitioned into five
folds of two sites each, so that all ten sites are held out exactly once. For each fold the
weights are fitted on the remaining sites and chi^2 is evaluated on the withheld restraints.

**Why grouped, not random.** Restraints sharing a spin label are strongly correlated —
neighbouring residues report almost the same probe distance — so a random split would put
near-duplicate measurements in both partitions and bias held-out error downward. Grouping
by site is cluster-based splitting, and it asks the meaningful question: does the ensemble
predict a spin-label experiment that was never run?

| Fold | Held-out sites | n | Held-out | All-data fit | Uniform prior | Optimism | Gain vs uniform |
|---|---|---|---|---|---|---|---|
| 1 | 15, 72 | 773 | 29.62 | 21.25 | 50.40 | +39.4% | +41.2% |
| 2 | 125, 322 | 748 | 12.85 | 10.03 | 27.41 | +28.1% | +53.1% |
| 3 | 178, 352 | 765 | 19.02 | 13.72 | 31.43 | +38.6% | +39.5% |
| 4 | 239, 384 | 761 | 19.40 | 14.24 | 36.46 | +36.2% | +46.8% |
| 5 | 256, 416 | 763 | 5.93 | 5.39 | 13.54 | +10.1% | +56.2% |
| **Mean** | | | **17.36** | **12.92** | **31.85** | **+30.5%** | **+47.4%** |

*All-data fit* is chi^2 on the same withheld restraints under weights fitted to all data;
*uniform prior* is the unreweighted AlphaFold ensemble.

**Result.** Held-out chi^2 exceeds the all-data fit by 30.5% on average — the expected
in-sample optimism. Against the unreweighted prior it is 47.4% lower on average, and lower
in every fold (range 39.5–56.2%). The inferred weights therefore predict restraints from
spin labels never seen during fitting substantially better than no reweighting, which is
the evidence that the posterior captures transferable structural information rather than
fitting noise.

**Weight agreement.** Predictive accuracy does not by itself show that the same structures
are selected. Comparing each fold's posterior with the all-data fit:

| Fold | Held-out sites | top-10 | top-50 | Spearman (top 100) | JS distance | N_eff |
|---|---|---|---|---|---|---|
| 1 | 15, 72 | 5/10 | 29/50 | 0.477 | 0.420 | 60.8 |
| 2 | 125, 322 | 8/10 | 42/50 | 0.859 | 0.245 | 50.9 |
| 3 | 178, 352 | 8/10 | 33/50 | 0.658 | 0.334 | 27.6 |
| 4 | 239, 384 | 6/10 | 37/50 | 0.726 | 0.315 | 50.1 |
| 5 | 256, 416 | 8/10 | 42/50 | 0.911 | 0.114 | 28.9 |
| **Mean** | | **7.0/10** | **36.6/50** | **0.726** | **0.286** | **43.7** |

Top-10 and top-50 are the number of the all-data fit's highest-weight models that remain
in the fold's own top 10 and top 50; `N_eff` is the effective ensemble size,
`exp(-sum w log w)`, against 26.0 for the all-data fit.

About 70% of the top 10 and 73% of the top 50 survive any single fold, so the identity of
the high-weight sub-ensemble is largely preserved. Rank agreement within the top is only
moderate (Spearman 0.73), so ensemble-level quantities are better supported than the
ordering of individual models.

