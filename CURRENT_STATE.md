# CURRENT_STATE.md
## Project status
The AgPd basic rebuild has progressed through preprocessing, model construction, prior predictive checks, posterior inference, likelihood refinement, posterior observable diagnostics, residual diagnostics, and an initial PSIS-LOO comparison for Ag10Pd90.
**Current checkpoint:** 160 tests passing.
**Current focus:** make Ag10Pd90 post-processing permanent and reproducible before moving to other AgPd compositions.
**Development branch context:** active rebuild work is on `rebuild/from-scratch`, with Ag10Pd90 currently the most developed posterior-analysis case.
The current repository contains:
- AgPd basic-data preprocessing and processed-data contracts
- nonrectangular model-data indexing
- generic PyMC/PyTensor model infrastructure
- AgPd mechanism kernels
- prior profiles and prior-predictive checks
- iid and setup-aware log-rate likelihoods
- posterior sampling/reconstruction/log-likelihood support
- derived-observable maps and posterior observable diagnostics
- preliminary residual/physical diagnostics
Post-processing is not yet complete. Several useful calculations have only been run as temporary terminal diagnostics and now need to be promoted into repository code, saved numerical outputs, plotting scripts, and tests.

## Repository foundation
Completed:
- Clean `src/` package layout with editable installation.
- Immutable `data/raw/` policy.
- `data/processed/`, `scripts/`, `config/`, `results/`, and `tests/` organization.
- Pytest from repository root.
- Separation of preprocessing configuration, model configuration, priors, and mechanism registry.
- Actual observed condition combinations are represented rather than forcing rectangular concentration × pressure grids.
- Current tests: **179 passed**.
Current warnings:
- Existing `threadpoolctl` Intel/LLVM OpenMP warnings from the Windows environment.
- One `NumbaPerformanceWarning` from the tiny setup-centering matrix dot product during prior predictive testing. This is currently treated as performance-only and is not being changed unless it affects real fits.

## AgPd basic-data preprocessing
### Raw-data contract
Implemented and tested for:
- Materials: `Pd100`, `Ag10Pd90`, `Ag25Pd75`, `Ag50Pd50`, `Ag75Pd25`, `Ag90Pd10`
- KOH: 0.25, 0.50, 1.00 M
- CO mole fraction: 0.001, 0.01, 0.10, 1.00
- Replicates: A/B/C
- Potential reference: SHE
- Raw current representation: `ln(j)`, underlying `j` in µA/cm²_Pd
- AgPd raw curves: 1000 acquired points expected per curve
Raw ingestion preserves acquisition `point_index`, `ln_j_uA_cm2`, and reconstructed `j_uA_cm2`.

### Potential-grid processing
Implemented:
- Condition-by-condition replicate alignment.
- Increasing/decreasing monotonic sweeps handled explicitly.
- Non-monotonic sweeps rejected at model-grid construction.
- No extrapolation beyond measured replicate domains.
- PCHIP interpolation in `ln(j)`.
- Canonical AgPd analysis grid: 10 mV spacing, origin 0 V.
- `analysis_grid_index` distinct from acquisition `point_index`.
The 10 mV grid is the fitting/observable grid. It reduces raw oversampling but is not assumed to make neighboring residuals statistically independent.

### Rate normalization
For CO oxidation:
\[
r = \frac{j}{2(210)} = \frac{j}{420}\;\mathrm{s^{-1}}
\]
for `j` in µA/cm²_Pd.
Interpretation:
- TOF is per Pd-ECSA-equivalent site.
- It is not a turnover rate per total Ag + Pd surface atom.
- Increasing Pd-normalized TOF with Ag content is not by itself evidence for an electronic effect.

### Truncation
Current AgPd fitting-window rule:
- low-potential threshold \(r \ge 10^{-3}\;\mathrm{s^{-1}}\)
- all three replicates must satisfy the threshold
- threshold must remain satisfied through the condition-level mean-rate peak
- no high-potential truncation
- post-peak data retained even if rate later falls below threshold
Current processed counts:
- Full analysis rows: 8856
- Selected replicate rows: 8793
- Selected summary points: 2931
- Conditions truncated: 6

## Experimental design and derived observables
The CO-series pairing has now been confirmed:
- At fixed material and KOH, replicate A is one connected CO-pressure series; likewise B and C.
- Pairing key across CO: `(material, C_KOH_M, replicate)`.
- Different KOH concentrations are independent experimental setups; replicate labels do not pair across KOH.

### Transfer coefficient
\[
\alpha = \frac{RT}{F}\frac{d\ln r}{dE}
\]
using numerical differentiation on the processed potential grid.

### OH reaction order
`delta_OH` remains an across-KOH log-slope using 0.25, 0.50, and 1.00 M where available. Because KOH concentrations are independent setups, no replicate pairing is imposed across KOH.

### CO reaction order
The old uncertainty propagation based on replicate-averaged rates was replaced with paired replicate-specific adjacent CO orders:
\[
\delta_{\mathrm{CO},r}
=
\frac{\ln r_{r,H}-\ln r_{r,L}}
{\ln(y_H/y_L)}
\]
for replicate \(r\in\{A,B,C\}\).
The reported experimental `delta_CO` is the mean across paired A/B/C values and its uncertainty is the sample SD (`ddof=1`) across those three paired orders.
Implemented outputs:
- `AgPd_COOx_basic_delta_CO_replicates.parquet`
- `AgPd_COOx_basic_delta_CO.parquet`
Current counts:
- CO-order replicate rows: 6579
- CO-order summary rows: 2193
- \(6579=3\times2193\), confirming three paired replicate orders per summary point

## Generic model-data and observable architecture
### `model_data.py`
Creates canonical:
- `conditions`
- `model_points`
- `observations`
Properties:
- only actually observed condition combinations
- contiguous stable IDs
- explicit replicate observations
- condition-specific potential windows

### `model_inputs.py`
Creates PyMC/PyTensor-ready arrays for:
- material/condition/model-point indexing
- SHE potential
- log electrolyte concentration
- log CO mole fraction
- observation-to-model-point mapping
- replicate-resolved observed log rates
It also now supports setup-aware indexing:
- `setup_labels`
- `setup_material_index`
- `observation_setup_index`
- `setup_experiment_index`
- `setup_experiment_size`

### `observable_maps.py`
Fixed linear maps for model log-rate draws:
\[
q_k=\sum_j w_{kj}\ln r_j
\]
Implemented for:
- \(\alpha\)
- \(\delta_{\mathrm{OH}}\)
- adjacent \(\delta_{\mathrm{CO}}\)
These maps operate on nonrectangular model-point designs.

## Current likelihood
The preferred working likelihood is now the zero-sum setup-intercept log-rate likelihood.

For observation \(i\):
\[
\ln r_i^{\mathrm{obs}}
\sim
\mathcal N\left(
\ln r_{p(i)}^{\mathrm{model}} + b_{s(i)},
\sigma_{\mathrm{res},m(i)}
\right)
\]
where:
- \(p(i)\) is the model point
- \(s(i)\) is the setup `(material, electrolyte_concentration_M, replicate)`
- \(m(i)\) is the material

Residual scale:
\[
\sigma_{\mathrm{res},m}
\sim
\mathrm{LogNormal}(\ln 0.20,0.75)
\]

Setup scale:
\[
\sigma_{\mathrm{setup},m}
\sim
\mathrm{LogNormal}(\ln 0.10,0.75)
\]

The setup offsets are constrained to zero-sum within each `(material, electrolyte_concentration_M)` experiment:
\[
b_{k,A}+b_{k,B}+b_{k,C}=0
\]
This prevents nuisance offsets from shifting an entire KOH dataset and stealing mechanistic KOH dependence.
Implementation uses centered latent standard normals with the variance-preserving factor:
\[
b_s=
\sigma_{\mathrm{setup},m(s)}
\sqrt{\frac{n_g}{n_g-1}}
\left(z_s-\bar z_{g(s)}\right)
\]
For the current A/B/C groups, \(n_g=3\).
A prior-predictive test verifies exact zero-sum behavior to numerical precision.
The original iid likelihood remains available as a control comparison.

## AgPd mechanisms currently rebuilt
Registered Ag10Pd90 models:
- `BF`
- `BF_LH`
- `CO_BF_ER_LH`
Implemented chemistry includes:
- QEA Pd competitive CO/OH adsorption
- QEA Ag-OH adsorption
- BF, LH, ER pathways
- BF + LH
- BF + ER + LH
- analytic CO-SSA model for `CO_BF_ER_LH`
The CO-SSA quadratic solution remains analytic and stable; no iterative nonlinear solver is required.
Current conventions include:
\[
\Delta G(E)=\Delta G_0-nE
\]
and
\[
G^\ddagger(E)=G^\ddagger_0-\beta nE
\]
with energies in eV and potential in V.
Surface-composition assumptions remain mean-field:
- CO adsorbs only on Pd
- ER has no additional composition prefactor after Pd-ECSA normalization
- LH carries \(x_{\mathrm{Pd}}\)
- BF carries \(x_{\mathrm{Ag}}\)

## Priors
Ag10Pd90 prior profiles currently remain the legacy-tuned profiles used as a controlled bridge to the previous analysis.
They are explicitly not treated as mechanistically neutral.
The current workflow preserves them first, then evaluates posterior contraction, boundary behavior, and later prior sensitivity rather than changing implementation, likelihood, and priors simultaneously.
Only Ag10Pd90 prior/model profiles are currently rebuilt. Other alloy compositions must be extracted from their own source notebooks rather than inferred from Ag10Pd90.

## Ag10Pd90 posterior inference
All three setup-aware models have been fit with nutpie/numba, 4 chains, 1000 tune, 1000 retained draws, target acceptance 0.9.

### BF
- Wall time: 22.32 s
- Divergences: 0
- max R-hat: 1.000
- min bulk ESS: 580
- min tail ESS: 650
- \(\sigma_{\mathrm{res}}\): mean 0.4289
- \(\sigma_{\mathrm{setup}}\): mean 0.0756

### BF_LH
- Wall time: 23.88 s
- Divergences: 0
- max R-hat: 1.000
- min bulk ESS: 480
- min tail ESS: 570
- \(\sigma_{\mathrm{res}}\): mean 0.4249
- \(\sigma_{\mathrm{setup}}\): mean 0.0766

### CO_BF_ER_LH
- Wall time: 55.34 s
- Divergences: 0
- max R-hat: 1.000
- min bulk ESS: 150
- min tail ESS: 160
- \(\sigma_{\mathrm{res}}\): mean 0.4235
- \(\sigma_{\mathrm{setup}}\): mean 0.0749

The setup-aware likelihood substantially improved sampler geometry for `CO_BF_ER_LH` relative to its earlier iid fit, which had one divergence, max R-hat about 1.10, and poor minimum ESS.

## Setup-offset validation
For Ag10Pd90 BF, inferred setup offsets closely reproduce the empirical A/B/C offsets while satisfying exact zero-sum constraints.
Approximate empirical offsets:
- 0.25 M: A +0.037, B +0.064, C -0.101
- 0.50 M: A +0.019, B +0.047, C -0.066
- 1.00 M: A +0.112, B -0.077, C -0.035
Approximate BF posterior means:
- 0.25 M: A +0.031, B +0.054, C -0.085
- 0.50 M: A +0.017, B +0.040, C -0.057
- 1.00 M: A +0.096, B -0.067, C -0.030
Posterior sums within KOH are zero to ~\(10^{-16}\).
Conclusion: the setup component is behaving as intended and is not absorbing the overall KOH dependence.

## Posterior observable diagnostics
The corrected setup-aware likelihood does not materially change the mechanistic observable mismatch relative to iid fits.

### BF
Setup-aware:
- median \(|\Delta\alpha|\): 0.0333
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.1050
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2861
- median standardized `delta_CO` residual: 12.507
- experimental mean inside 95% posterior `delta_CO` interval: 2.0%

### BF_LH
Setup-aware:
- median \(|\Delta\alpha|\): 0.0305
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.1007
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2862
- median standardized `delta_CO` residual: 12.231
- experimental mean inside 95% posterior `delta_CO` interval: 2.8%

### CO_BF_ER_LH
Setup-aware:
- median \(|\Delta\alpha|\): 0.0281
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.0847
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2871
- median standardized `delta_CO` residual: 12.356
- experimental mean inside 95% posterior `delta_CO` interval: 2.3%

The common `delta_CO` mismatch is therefore not explained by the original iid treatment of paired CO-series data.

## Residual diagnostics
Current conditional residual:
\[
e_i=
\ln r_i^{\mathrm{obs}}
-
\left(
\ln r_i^{\mathrm{model}}+b_{s(i)}
\right)
\]

For the setup-aware fits:
- BF residual RMS: 0.4271
- BF_LH residual RMS: 0.4231
- CO_BF_ER_LH residual RMS: 0.4214
Median adjacent-potential residual correlation:
- BF: 0.993
- BF_LH: 0.985
- CO_BF_ER_LH: 0.985
This is a diagnostic of smooth residual structure, not a fitted AR(1) parameter and not proof that the experimental noise itself follows AR(1).

For BF_LH, a replicate decomposition was performed for every KOH × CO condition:
- shared residual = A/B/C mean residual curve
- replicate residual = each replicate minus that shared curve
The median fraction of squared residual structure attributable to the shared curve was:
\[
0.984
\]
All 12 KOH × CO conditions had shared fractions above 0.928.
Interpretation: most remaining discrepancy is common across replicates and is therefore not primarily replicate-specific scatter. The fitted \(\sigma_{\mathrm{res}}\approx0.42\) should not be interpreted as pure experimental noise; it is absorbing substantial model discrepancy.

Current decision:
- do not immediately escalate to a flexible correlated-error/GP/random-slope likelihood
- do not use a correlation model merely to rename reproducible mechanistic discrepancy as noise
- retain the zero-sum setup-intercept likelihood as the working likelihood for now

## Physical posterior checks
Across all three setup-aware models:
- coverages remain in [0,1]
- Pd site balance is satisfied to machine precision
- Ag site balance is satisfied to machine precision
- pathway fractions, where applicable, remain in [0,1] and sum to one to machine precision
No obvious nonphysical coverage/site-balance pathology is driving the fits.

Selected pathway behavior:
### BF_LH
Pooled posterior pathway fractions:
- BF median ~0.998
- LH median ~0.0017
- LH 97.5% quantile ~0.409
The pooled values mix conditions and draws and should not be interpreted as a single global pathway fraction. Pointwise pathway plots are still required.

### CO_BF_ER_LH
Pooled posterior pathway fractions:
- BF median ~0.482
- ER median ~0.461
- LH median ~0.0022
Again, pointwise condition-resolved analysis is required before mechanistic interpretation.
`Gact1_0` is comparatively weakly constrained in the large model, with posterior/prior SD ratio ~0.53.

## Initial PSIS-LOO comparison
Observation-wise PSIS-LOO has been run on the three converged setup-aware Ag10Pd90 posteriors.

Individual ELPD:
- BF: -824.71, SE 18.54, \(p_{\mathrm{loo}}=10.67\), max Pareto-k 0.319
- BF_LH: -811.54, SE 17.71, \(p_{\mathrm{loo}}=11.06\), max Pareto-k 0.175
- CO_BF_ER_LH: -807.39, SE 17.75, \(p_{\mathrm{loo}}=12.09\), max Pareto-k 0.154

All 1431 pointwise Pareto-k values are below 0.5 for every model.
Approximate ELPD differences:
- BF_LH - BF: +13.17
- CO_BF_ER_LH - BF_LH: +4.15
- CO_BF_ER_LH - BF: +17.32
`az.compare` ranked:
1. `CO_BF_ER_LH`
2. `BF_LH`
3. `BF`
with approximate difference uncertainty of ~2 ELPD for the large model versus BF_LH and ~4 for the large model versus BF.
Stacking assigned weight 1.0 to `CO_BF_ER_LH`; this is a predictive mixture weight, not a posterior probability that the mechanism is true.

Important caveat:
This is ordinary observation-wise LOO. When one point is held out, the other observations from the same experimental setup remain in the fit and continue to inform the setup offset. It therefore answers a within-characterized-setup prediction question, not prediction of a completely new experimental setup. The smooth shared residual structure also means LOO ranking must not be promoted to proof of mechanism.

## Current scientific interpretation
### Mathematically established
- AgPd preprocessing, normalization, and nonrectangular model indexing.
- Paired CO-order construction.
- QEA site balances.
- Stable parallel-pathway log-rate construction.
- Analytic CO-SSA solution.
- Zero-sum setup-intercept likelihood construction.
- Exact setup zero-sum behavior.
- Posterior coverage/site/pathway balance checks.
- Stable PSIS diagnostics for the current observation-wise LOO calculation.

### Supported statistical conclusions
- The original iid likelihood misses real persistent A/B/C setup structure across CO series.
- The zero-sum setup-intercept model captures those setup differences without shifting whole KOH experiments.
- BF is predictively inferior to BF_LH and CO_BF_ER_LH under the current setup-aware likelihood.
- The large model has the best current observation-wise ELPD, but its advantage over BF_LH is much smaller than either model's advantage over BF.
- The current posterior residual scale around 0.42 is dominated by reproducible model discrepancy rather than simple replicate noise.

### Supported mechanistic interpretation
- The current BF formulation is inadequate for the Ag10Pd90 rate surface.
- Adding LH and/or the larger CO/BF/ER/LH structure improves some potential/OH behavior and predictive performance.
- All three current ideal mean-field models retain essentially the same severe CO-order mismatch, \(|\Delta\delta_{\mathrm{CO}}|\approx0.286\), after the experimental pairing and likelihood corrections.
- The current mechanism family is therefore missing physical behavior required to reproduce the measured CO-pressure dependence.

### Plausible but unresolved
- Lateral interactions or other non-ideal surface physics could contribute to the CO-order behavior.
- This is intentionally deferred; no lateral-interaction model is being developed at the current stage.
- The relative mechanistic roles of BF, LH, and ER remain unresolved because pathway tradeoffs and shared model deficiency remain.

### Insufficient information to conclude
- That `CO_BF_ER_LH` is the true mechanism merely because it has the best current ELPD.
- That the ER pathway is mechanistically established.
- That the small formal posterior SDs represent calibrated physical parameter uncertainty while the residual likelihood remains misspecified.
- That Ag10Pd90 behavior generalizes to other AgPd compositions.

## Post-processing status
Post-processing is now the immediate repository-development priority.
Several diagnostics were first run as disposable terminal probes. Any result intended to remain part of the workflow must now be moved into permanent code with saved numerical outputs and tests.

Required permanent post-processing components:
- posterior parameter summaries
- prior-versus-posterior overlays and contraction
- sampler diagnostics, energy/BFMI, ESS/R-hat, selected pair/ridge plots
- mechanism-only rate fits
- setup-adjusted conditional rate fits
- posterior predictive rate/log-rate plots
- residual grids and shared-vs-replicate residual decomposition
- experimental vs posterior \(\alpha\)
- experimental vs posterior \(\delta_{\mathrm{OH}}\)
- experimental vs posterior paired \(\delta_{\mathrm{CO}}\)
- coverages versus potential
- pointwise BF/LH/ER pathway fractions
- PSIS-LOO tables
- Pareto-k plots
- pointwise LOO/ELPD grids
- pointwise ELPD-difference plots between models
- LOO-PIT and other useful predictive-calibration diagnostics
- permanent saved numerical products for all derived quantities
- rebuilt DRC calculations and DRC plots

Architecture preference:
\[
\text{posterior}
\rightarrow
\text{saved derived numerical result}
\rightarrow
\text{plot}
\]
Plotting functions should not silently perform expensive or scientifically important calculations that are otherwise unavailable.

Suggested eventual interface:
```text
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood setup_intercept
python scripts/compare_agpd_models.py --material Ag10Pd90 --likelihood setup_intercept
```
with outputs organized into persistent tables/derived arrays/figures rather than terminal-only results.

## DRC status
DRC infrastructure has not yet been rebuilt against the new model architecture.
Future DRC work must:
- perturb thermodynamically meaningful quantities consistently
- preserve thermodynamic relationships between forward/reverse quantities
- operate over posterior draws
- save condition-resolved DRC arrays before plotting
- verify expected DRC sum behavior where the formulation requires it
- support multiple simultaneously rate-controlling steps rather than forcing a single RDS interpretation

## Known unresolved issues
1. **Post-processing is incomplete.**
   Existing exploratory diagnostics need to become permanent scripts/modules/tests and saved outputs.
2. **Residual likelihood remains imperfect.**
   Zero-sum setup effects are handled, but smooth shared residual structure remains. No more flexible correlated-error model is being added at present because much of the structure appears to be reproducible model discrepancy.
3. **Ag10Pd90 priors are legacy-tuned.**
   Prior-sensitivity analyses remain required before strong parameter/pathway interpretation.
4. **DRC has not yet been rebuilt.**
5. **Only Ag10Pd90 model/prior profiles are rebuilt.**
   Other AgPd compositions must be reconstructed from their own source notebooks/configuration.
6. **Observation-wise LOO is not new-setup LOO.**
   Grouped predictive questions may later require grouped/leave-one-setup-out validation.
7. **Lateral interactions are deliberately deferred.**
   They remain a possible future physical extension, not a current implementation target.
8. **PtRu remains deferred.**
   Current implementation work is focused on completing and validating AgPd basic first.

## Immediate plan
1. Perform a repository-wide housekeeping review before adding more post-processing infrastructure.
2. Check repository organization, source-of-truth/config duplication, obsolete exploratory scripts, path conventions, naming consistency, unused/dead utilities, result-directory conventions, and test coverage.
3. Promote trusted temporary diagnostics into reusable `src/mkm/...` functions and permanent scripts.
4. Build the complete Ag10Pd90 post-processing pipeline and plotting suite.
5. Rebuild thermodynamically consistent DRC calculation/storage/plotting.
6. Re-run the permanent post-processing workflow for BF, BF_LH, and CO_BF_ER_LH.
7. Only after Ag10Pd90 post-processing is complete, extract model/prior profiles and repeat the validated workflow for the other AgPd compositions.

## Current checkpoint summary
The key result of the current phase is not simply a model ranking.
The rebuild has established that:
- the paired CO experimental design matters and is now represented correctly
- zero-sum A/B/C setup offsets are real and recoverable
- those offsets do not explain the large Ag10Pd90 discrepancy
- residual discrepancy is overwhelmingly shared across replicates
- all three current mechanisms severely miss the paired CO-order behavior
- BF_LH remains predictively better than BF
- the larger CO_BF_ER_LH model is currently best by observation-wise ELPD, but this does not establish it as the true mechanism
- post-processing, model comparison visualization, and DRC are the next major engineering/scientific tasks
