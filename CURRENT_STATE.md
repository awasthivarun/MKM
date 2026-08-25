# CURRENT_STATE.md

## Project status

The AgPd basic rebuild now supports both individual-material and multi-material composition inference:

```text
preprocessing
→ model construction
→ posterior inference
→ sampler/physical diagnostics
→ experimental observable comparison
→ PSIS-LOO and LOO-PIT
→ model comparison
→ transition-state DRC for individual-material fits
```

Current scientific scope: AgPd CO oxidation in basic media across

- `Pd100`
- `Ag10Pd90`
- `Ag25Pd75`
- `Ag50Pd50`
- `Ag75Pd25`
- `Ag90Pd10`

Current working likelihood: material-specific IID Normal residuals in log-rate space.

The zero-sum setup-intercept likelihood remains implemented for comparison/history but is not the current working likelihood. Strong residual correlation along potential remains present under IID and is intentionally deferred while the mechanistic mean function is developed.

## AgPd basic preprocessing

### Experimental contract

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
r=\frac{j}{420}\;\mathrm{s^{-1}}
\]

for `j` in µA/cm²_Pd.

This is a Pd-ECSA-normalized rate, not a per-total-surface-atom rate.

For the AgPd mechanisms, explicit `Ag_fraction` and `Pd_fraction` factors represent random-mixing ensemble/neighbor probabilities relative to a Pd-centered rate normalization. They are not additional ECSA normalization factors.

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

Within fixed material/KOH, replicate A across all CO pressures is one connected series; likewise B and C. Pairing resets across KOH concentrations.

Pairing key:

```text
(material, C_KOH_M, replicate)
```

Replicate-specific adjacent CO order:

\[
\delta_{\mathrm{CO},r}
=
\frac{\ln r_{r,H}-\ln r_{r,L}}{\ln(y_H/y_L)}.
\]

The summary uses the mean and sample SD across A/B/C.

## Generic data/model architecture

`model_data.py` creates canonical `conditions`, `model_points`, and `observations`. Only observed combinations are represented.

`model_inputs.py` builds material/condition/model-point arrays, observed log rates, observation-to-model-point mappings, setup indices, and mechanism inputs.

Fixed observable maps produce:

- \(\alpha\)
- \(\delta_{\mathrm{OH}}\)
- adjacent \(\delta_{\mathrm{CO}}\)

from posterior model log-rate draws.

## Current likelihood

The working model is

\[
\ln r_i^{\mathrm{obs}}
\sim
\mathcal N\!\left(\ln r_i^{\mathrm{model}},\sigma_{m(i)}^2\right),
\]

with one `sigma_ln_rate_material` per material.

The residual scale is interpreted as an effective combination of experimental scatter, unmodeled setup variation, and model discrepancy. It is not a pure measurement standard deviation.

Residuals remain strongly correlated between adjacent potential points. Therefore IID posterior HDIs can be narrower than calibrated physical uncertainty. This does not prevent the IID model from being used to develop and compare mechanistic mean functions.

## Mechanisms

Registered models:

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
- exact detailed balance for reversible CO adsorption/desorption

Potential conventions:

\[
\Delta G(E)=\Delta G_0-nE,
\]

\[
G^\ddagger(E)=G^\ddagger_0-\beta nE.
\]

For the reversible CO step,

\[
\ln k_{-1}=\ln k_1-\ln K_1,
\]

so the reverse reaction tracks the forward barrier and reaction free energy exactly.

## Composition inference

Composition workflows use all six materials in a single fit.

Available parameterizations:

- `shared`
- `linear_xAg`

The linear parameterization uses

\[
p(x_{\mathrm{Ag}})=p_{0.5}+s_p(x_{\mathrm{Ag}}-0.5).
\]

Current `CO_BF_ER_LH` slopes:

- `deltaG1_0_xAg_slope`
- `deltaG4_0_xAg_slope`
- `deltaG5_0_xAg_slope`
- `Gact2_BF_0_xAg_slope`
- `Gact2_ER_0_xAg_slope`

The slope priors are regularizing priors intended to exclude remote, implausible multimodal branches. The numerical bounds are not treated as universal physical bounds on effective energies at every composition.

### Current converged `linear_xAg / iid / CO_BF_ER_LH` checkpoint

Sampler:

- 0 divergences
- max R-hat: 1.0059
- min bulk ESS: 621
- min tail ESS: 881

Posterior slope medians and 95% HDIs:

| parameter | median | 95% HDI |
| --- | ---: | ---: |
| `deltaG1_0_xAg_slope` | 0.04547 | [0.04305, 0.04789] |
| `deltaG4_0_xAg_slope` | -0.05370 | [-0.06360, -0.04419] |
| `deltaG5_0_xAg_slope` | -0.03005 | [-0.03681, -0.02270] |
| `Gact2_BF_0_xAg_slope` | 0.04056 | [0.03522, 0.04656] |
| `Gact2_ER_0_xAg_slope` | -0.02482 | [-0.03214, -0.01723] |

These are statistically well resolved within the current IID model. Their narrow HDIs should not be interpreted as fully calibrated physical uncertainty because the residual process is strongly correlated along potential.

The composition-induced energy changes are on the scale of tens of meV across the measured composition range. No hard BEP relationship is imposed; only exact detailed-balance relationships are enforced.

## Current material behavior

The converged linear composition model improves the description of much of the alloy series but Ag10Pd90 remains a major weakness.

Current IID material residual scales for the converged linear model are approximately:

| material | median `sigma_ln_rate_material` |
| --- | ---: |
| Ag10Pd90 | 1.332 |
| Ag25Pd75 | 0.455 |
| Ag50Pd50 | 0.361 |
| Ag75Pd25 | 0.348 |
| Ag90Pd10 | 0.431 |
| Pd100 | 0.343 |

Ag10Pd90 therefore deserves explicit held-out-material testing rather than being interpreted from observation-wise LOO alone.

## Experimental observables and residual structure

The model continues to reproduce some rate-surface behavior while showing strong mismatch in experimental observables, particularly CO reaction order.

Residuals remain highly correlated along potential within experimental curves. Observation-wise PSIS-LOO is therefore useful for locating where a model succeeds or fails, but it does not represent prediction of a new curve or a new material.

## Predictive-validation plan

Two refit-based validation levels are now preferred:

### LOCO — leave one condition out

Hold out one material-specific `(KOH, CO)` condition and all three A/B/C replicate curves at that condition.

Example:

```text
Ag50Pd50, 0.5 M KOH, 10% CO
```

All potential points from replicates A, B, and C are excluded from fitting.

This tests whether the mechanism can predict an unseen experimental condition within a known material.

### LOMO — leave one material out

Hold out one material/composition entirely.

This directly tests whether `linear_xAg` energetics generalize to an unseen alloy composition. Ag10Pd90 is a particularly informative target because it is currently poorly described by the composition fit.

Grouped summaries of existing pointwise ELPD are not a current priority because the pointwise ELPD plots already resolve the behavior by condition and replicate. LOCO/LOMO answer stronger predictive questions.

## Permanent postprocessing

Individual-material command:

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood iid
```

Composition command:

```powershell
python scripts/postprocess_agpd_composition_posterior.py CO_BF_ER_LH --composition-model linear_xAg --likelihood iid
```

Persistent products include:

- sampler diagnostics
- posterior parameter summaries and contraction tables
- posterior predictive rate grids
- residual grids and residual-structure summaries
- coverages and pathway fractions
- alpha/OH-order/CO-order comparisons
- pointwise PSIS-LOO
- Pareto-k
- LOO-PIT diagnostics
- machine-readable tables and parquet products

For composition fits, `posterior_parameter_summary.csv` includes the fitted reference parameters and composition slopes.

## Transition-state DRC

Current definition:

\[
X_{\mathrm{TS},j}
=
-k_BT\frac{\partial\ln r}{\partial G^\ddagger_j}.
\]

The current DRC implementation is transition-state-only and was validated for the individual-material mechanisms using sum rules and finite-difference step checks.

Composition-specific DRC is not yet implemented. For `linear_xAg`, the correct future workflow is to construct the effective transition-state energy at each composition and perturb that composition-specific transition state.

## Scientific interpretation

### Mathematically established

- preprocessing and normalization contracts
- paired CO-order calculation
- nonrectangular model indexing
- QEA site balances
- analytic CO-SSA solution
- exact forward/reverse thermodynamic relationship for CO adsorption/desorption
- stable log-space pathway combination
- composition transformation \(p(x)=p_{0.5}+s_p(x-0.5)\)

### Supported statistical conclusions

- the current five-slope linear composition fit is numerically converged
- shared energetics create substantial material-dependent mismatch
- composition-dependent energetics improve the description of several materials
- Ag10Pd90 remains poorly described
- residual correlation along potential is strong and violates the ideal IID-error picture

### Supported mechanistic interpretation

- pure BF is inadequate for the full AgPd problem
- ER-containing chemistry is needed to describe Pd100 within the current mechanism family
- energetic quantities vary systematically with composition within the current model

### Plausible but unresolved

- whether the approximately linear energetic trends extrapolate predictively to an unseen composition
- detailed BF/ER/LH partitioning
- causes of the persistent CO-order mismatch
- whether a future correlated likelihood materially changes parameter uncertainty

### Insufficient information to conclude

- that the largest model is the true mechanism
- that narrow IID HDIs represent calibrated physical uncertainty
- that observation-wise LOO demonstrates prediction of new curves or materials

## Current development priorities

1. ensure multi-material sampler diagnostics include all material-specific noise parameters
2. update documentation to the IID/composition workflow
3. allow direct `shared` versus `linear_xAg` model comparison
4. remove silent legacy posterior-path fallback
5. add composition-aware prior predictive checks
6. add posterior energy-vs-composition products
7. add composition-specific transition-state DRC
8. implement LOCO and LOMO validation workflows
9. consolidate model-comparison LOO handling
10. split composition postprocessing orchestration after correctness changes
