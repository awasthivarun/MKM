# CURRENT_STATE.md

## Project status

The AgPd basic rebuild now contains a complete first-generation Ag10Pd90 workflow:

```text
preprocessing
→ model construction
→ prior predictive checks
→ posterior inference
→ sampler/physical diagnostics
→ experimental observable comparison
→ single-model PSIS-LOO and LOO-PIT
→ multi-model ELPD comparison
→ posterior transition-state DRC
```

**Validated checkpoint:** **192 tests passed, 7 known warnings**

**Current scientific scope:** AgPd CO oxidation in basic media, with Ag10Pd90 as the first fully rebuilt posterior-analysis case.

**Current engineering focus:** repository consolidation, reproducible environment specification, removal of workflow duplication, and preparation for additional materials/models.

## Repository foundation

Implemented:

- `src/` package layout and editable installation
- immutable raw-data policy
- dataset-specific preprocessing configuration
- separate model/likelihood/prior configuration
- nonrectangular model-data architecture
- explicit replicate observations
- zero-sum setup indexing
- reusable mechanism registry and PyMC assembly
- permanent postprocessing package
- canonical posterior/model-comparison/DRC scripts
- 192-test suite

Known warnings:

- seven non-failing environment/runtime warnings in the validated Windows environment
- the main recurring warning is Intel and LLVM OpenMP runtimes coexisting during short posterior tests
- warnings are not currently evidence of failed calculations, but the environment rebuild should recheck them

## Validated environment

Current working stack:

- Python 3.13.15
- PyMC 6.3.1
- PyTensor 3.3.0
- nutpie 0.16.11
- NumPy 2.4.6
- SciPy 1.18.0
- Numba 0.66.0
- pandas 3.0.5
- xarray 2026.7.0
- ArviZ split stack 1.3.0
- matplotlib 3.11.1
- pytest 9.1.1

`pyproject.toml` and `environment.yml` now define the intended package and development environment.

## AgPd basic preprocessing

### Experimental contract

- Materials:
  - `Pd100`
  - `Ag10Pd90`
  - `Ag25Pd75`
  - `Ag50Pd50`
  - `Ag75Pd25`
  - `Ag90Pd10`
- KOH: 0.25, 0.50, 1.00 M
- CO mole fraction: 0.001, 0.01, 0.10, 1.00
- Replicates: A/B/C
- Potential reference: SHE
- Raw current: `ln(j)`, underlying `j` in µA/cm²_Pd
- Expected raw AgPd curve length: 1000 points

### Potential processing

- monotonic sweeps handled explicitly
- non-monotonic sweeps rejected
- PCHIP interpolation in `ln(j)`
- no extrapolation
- canonical 10 mV analysis grid
- acquisition `point_index` remains distinct from `analysis_grid_index`

### Rate normalization

\[
r = \frac{j}{2(210)} = \frac{j}{420}\;\mathrm{s^{-1}}
\]

for `j` in µA/cm²_Pd.

This is a Pd-ECSA-normalized rate, not a per-total-surface-atom rate.

### Truncation

- low-potential threshold \(r\ge10^{-3}\;\mathrm{s^{-1}}\)
- all three replicates must satisfy it
- threshold must persist to the condition-level mean-rate peak
- no high-potential truncation
- post-peak points retained

Processed counts:

- full analysis rows: 8856
- selected replicate rows: 8793
- selected summary points: 2931
- truncated conditions: 6

## Paired CO-series design and reaction order

Confirmed design:

- within fixed material/KOH, replicate A across all CO pressures is one connected series
- likewise B and C
- pairing resets across KOH concentrations

Pairing key:

```text
(material, C_KOH_M, replicate)
```

Replicate-specific adjacent CO order:

\[
\delta_{\mathrm{CO},r}
=
\frac{\ln r_{r,H}-\ln r_{r,L}}{\ln(y_H/y_L)}
\]

The summary uses the mean and sample SD across A/B/C.

Outputs:

- paired replicate CO-order rows: 6579
- summarized CO-order rows: 2193

## Generic data/model architecture

### Model data

`model_data.py` creates:

- `conditions`
- `model_points`
- `observations`

Only actually observed combinations are represented.

### Model inputs

`model_inputs.py` builds:

- material/condition/model-point arrays
- observed log-rate arrays
- observation-to-model-point mapping
- setup labels and indices
- zero-sum setup-experiment indices
- model-point mechanism inputs

### Observable maps

Fixed linear maps produce:

- \(\alpha\)
- \(\delta_{\mathrm{OH}}\)
- adjacent \(\delta_{\mathrm{CO}}\)

from model log-rate draws on nonrectangular designs.

## Current likelihood

Preferred likelihood:

\[
\ln r_i^{\mathrm{obs}}
\sim
\mathcal N\left(
\ln r_{p(i)}^{\mathrm{model}}+b_{s(i)},
\sigma_{\mathrm{res},m(i)}
\right)
\]

with setup:

```text
(material, electrolyte_concentration_M, replicate)
```

and zero-sum constraint within each material/KOH experiment:

\[
b_{k,A}+b_{k,B}+b_{k,C}=0.
\]

Priors:

\[
\sigma_{\mathrm{res},m}\sim\mathrm{LogNormal}(\ln0.20,0.75)
\]

\[
\sigma_{\mathrm{setup},m}\sim\mathrm{LogNormal}(\ln0.10,0.75).
\]

The iid likelihood remains available as a control.

## Mechanisms

Registered Ag10Pd90 models:

- `BF`
- `BF_LH`
- `CO_BF_ER_LH`

Implemented components:

- Pd competitive CO/OH QEA
- Ag-OH QEA
- BF, LH, ER pathways
- stable log-sum-exp pathway addition
- analytic CO-SSA quadratic solution
- mean-field Ag/Pd neighbor probabilities

Current conventions:

\[
\Delta G(E)=\Delta G_0-nE
\]

\[
G^\ddagger(E)=G^\ddagger_0-\beta nE.
\]

Only Ag10Pd90 prior profiles are rebuilt. They remain legacy-tuned profiles and are not treated as neutral physical priors.

## Posterior inference

All setup-aware fits used:

- nutpie / numba
- 4 chains
- 1000 tune
- 1000 retained draws
- target acceptance 0.9

### BF

- 0 divergences
- max R-hat approximately 1.00
- min bulk ESS approximately 580
- min tail ESS approximately 650
- residual log-rate scale approximately 0.429
- setup scale approximately 0.076

### BF_LH

- 0 divergences
- max R-hat approximately 1.006 in current ArviZ diagnostics
- min bulk ESS approximately 476
- min tail ESS approximately 567
- residual log-rate scale approximately 0.425
- setup scale approximately 0.077

### CO_BF_ER_LH

- 0 divergences in the setup-aware fit
- max R-hat approximately 1.00
- min bulk ESS approximately 150
- min tail ESS approximately 160
- residual log-rate scale approximately 0.423
- setup scale approximately 0.075

The setup-aware likelihood materially improved the large model's sampler geometry relative to its earlier iid fit.

## Setup validation

The fitted setup offsets reproduce the directly observed A/B/C pattern while summing to zero within each KOH condition to numerical precision.

Interpretation:

- persistent setup effects are real
- the setup model represents them correctly
- setup effects do not explain the large model-data discrepancy

## Experimental observables

Setup-aware posterior mismatch:

### BF

- median \(|\Delta\alpha|\): 0.0333
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.1050
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2861
- median standardized CO-order residual: 12.507

### BF_LH

- median \(|\Delta\alpha|\): 0.0305
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.1007
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2862
- median standardized CO-order residual: 12.231

### CO_BF_ER_LH

- median \(|\Delta\alpha|\): 0.0281
- median \(|\Delta\delta_{\mathrm{OH}}|\): 0.0847
- median \(|\Delta\delta_{\mathrm{CO}}|\): 0.2871
- median standardized CO-order residual: 12.356

The paired-data and likelihood corrections do not repair the CO-order failure.

## Residual diagnostics

Conditional residual:

\[
e_i=
\ln r_i^{\mathrm{obs}}
-
\left(\ln r_i^{\mathrm{model}}+b_{s(i)}\right).
\]

Residual RMS:

- BF: 0.4271
- BF_LH: 0.4231
- CO_BF_ER_LH: 0.4214

Median adjacent-potential residual correlation:

- BF: 0.993
- BF_LH: 0.985
- CO_BF_ER_LH: 0.985

For BF_LH, the median fraction of squared residual structure shared across A/B/C is:

\[
0.984.
\]

Therefore, the fitted residual scale around 0.42 is primarily a model-discrepancy scale, not pure experimental noise.

Current decision:

- retain the zero-sum setup likelihood
- do not add a flexible discrepancy process merely to absorb reproducible mechanistic failure
- revisit correlated/grouped error only for a clearly defined predictive question

## Physical posterior checks

Across current models:

- coverages remain in \([0,1]\)
- Pd site balance holds to machine precision
- Ag site balance holds to machine precision
- pathway fractions remain bounded and sum to one

No obvious nonphysical coverage pathology explains the poor CO-order behavior.

## Permanent postprocessing

Per-model canonical command:

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood setup_intercept
```

Persistent products include:

- parameter posterior plots and contraction tables
- trace/rank/energy/pair diagnostics
- posterior predictive rate grids
- conditional residual grids
- shared-vs-replicate residual tables
- coverages and pathway fractions
- alpha/OH-order/CO-order comparisons
- pointwise PSIS-LOO
- Pareto-k
- LOO-PIT ECDF/coverage/condition grids
- machine-readable tables and parquet products

Single-model LOO/calibration products belong with the model.

Multi-model products belong under the model-comparison directory.

## PSIS-LOO and model comparison

Individual observation-wise ELPD:

- BF: -824.71; SE 18.54; \(p_{\mathrm{loo}}=10.67\); max \(k=0.319\)
- BF_LH: -811.54; SE 17.71; \(p_{\mathrm{loo}}=11.06\); max \(k=0.175\)
- CO_BF_ER_LH: -807.39; SE 17.75; \(p_{\mathrm{loo}}=12.09\); max \(k=0.154\)

All 1431 Pareto-k values are below 0.5.

Approximate differences:

- BF_LH - BF: +13.17
- CO_BF_ER_LH - BF_LH: +4.15
- CO_BF_ER_LH - BF: +17.32

Predictive ranking:

1. `CO_BF_ER_LH`
2. `BF_LH`
3. `BF`

Stacking weights are predictive mixture weights, not mechanism probabilities.

LOO scope:

- observation-wise conditional prediction
- other observations from the same setup remain available
- not leave-one-setup-out prediction

### BF_LH LOO-PIT checkpoint

- mean: 0.4979
- median: 0.4682
- SD: 0.3113
- Uniform reference SD: 0.2887
- outside 0.05–0.95: 7.41%

These scalar values do not override the strong condition-resolved residual/PIT structure.

## Transition-state DRC

Definition:

\[
X_{\mathrm{TS},j}
=
-k_BT
\frac{\partial\ln r}{\partial G^\ddagger_j}.
\]

Calculation:

- central finite differences in transition-state energy
- all current thermodynamic relationships preserved by the existing mechanism parameterization
- posterior-draw and model-point resolved
- draws saved before plotting

### BF validation

- one control
- \(X_{\mathrm{BF}}=1\)
- max sum-rule error approximately \(2.7\times10^{-13}\)
- half-step difference approximately \(4.5\times10^{-13}\)

### BF_LH validation

- BF and LH controls
- sum-rule max error approximately \(4.4\times10^{-13}\)
- max half-step difference approximately \(1.9\times10^{-7}\)

### CO_BF_ER_LH validation

Controls:

- CO adsorption/desorption transition state
- BF
- ER
- LH

Results:

- max sum-rule error approximately \(5.8\times10^{-7}\)
- max half-step difference approximately \(2.5\times10^{-7}\)
- observed DRC range approximately -0.00244 to 0.99998
- no non-finite values

The small negative DRC values exceed the finite-difference error scale and are mathematically possible in the coupled SSA mechanism.

Only transition-state DRC is currently implemented.

Intermediate thermodynamic rate control is deferred until the held-fixed state/transition-state energy convention is explicitly defined.

## Scientific interpretation

### Mathematically established

- preprocessing and normalization contracts
- paired CO-order calculation
- nonrectangular model indexing
- QEA balances
- analytic CO-SSA solution
- zero-sum setup likelihood
- posterior coverage/site/pathway balances
- stable PSIS diagnostics
- transition-state DRC sum rules and step convergence

### Supported statistical conclusions

- persistent A/B/C setup effects exist and are modeled correctly
- BF is predictively worse than BF_LH and the larger model
- the large model has only a modest ELPD advantage over BF_LH
- most residual discrepancy is shared across replicates
- residual scale should not be interpreted as pure experimental noise

### Supported mechanistic interpretation

- pure BF is inadequate for the Ag10Pd90 rate surface
- extra pathway flexibility improves some potential/OH behavior
- the current ideal mean-field mechanism family fails the measured CO-pressure dependence

### Plausible but unresolved

- lateral/nonideal interactions may contribute to the CO-order behavior
- BF/ER partition remains only partially identifiable
- multiple transition states may control different condition regions

### Insufficient information to conclude

- that the largest model is the true mechanism
- that ER is mechanistically established
- that formal narrow parameter intervals are fully calibrated physical uncertainty
- that Ag10Pd90 conclusions generalize across AgPd compositions

## Current organizational bottlenecks

1. canonical scripts hard-code `Ag10Pd90`
2. path/config/context construction is duplicated across scripts
3. posterior orchestration remains approximately 457 lines
4. plotting remains a single approximately 634-line module
5. pointwise outputs, balances, and DRC controls are not centralized in model metadata
6. fit provenance/run manifests are not yet written
7. generated-results versioning policy is not explicit
8. CLI integration coverage remains limited
9. only Ag10Pd90 prior profiles are rebuilt

## Immediate plan

1. adopt the reviewed `pyproject.toml` and `environment.yml`
2. centralize paths and AgPd run context
3. add `--material` to canonical scripts
4. add run metadata/config/data hashes to fits
5. split posterior workflow orchestration
6. split plotting by domain
7. centralize model metadata
8. add CLI integration tests
9. perform prior-sensitivity analysis
10. extract/rebuild prior profiles for remaining AgPd materials
