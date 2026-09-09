# CURRENT_STATE

Documentation checkpoint: `d1b0bcedce7628f7af4ade4534e6c7cd1fee9d12` on `rebuild/from-scratch`.

This file records the **software/scientific workflow state**, not a frozen list of posterior results. Numerical conclusions should be taken from the current `results/` products rather than copied here unless they have become durable project knowledge.

## Active scientific scope

The most developed system is CO oxidation on AgPd in basic media at 293.15 K for:

- `Pd100`
- `Ag10Pd90`
- `Ag25Pd75`
- `Ag50Pd50`
- `Ag75Pd25`
- `Ag90Pd10`

The surface composition is currently taken equal to bulk composition and random mixing is assumed.

## End-to-end workflow implemented

```text
raw workbooks
  -> standardized replicate data
  -> interpolated/truncated analysis-grid data
  -> model conditions / model points / observations
  -> prior predictive checks
  -> individual or all-material posterior inference
  -> sampler + parameter + physical diagnostics
  -> rate/residual/pathway/coverage products
  -> alpha / OH-order / CO-order comparison
  -> PSIS-LOO + LOO-PIT
  -> multi-model comparison
  -> transition-state DRC
  -> composition-parameter trend plots
  -> LOCO / LOMO refit validation
```

## Active likelihood

The only active likelihood is Normal in linear rate space:

$$
r_i^{\mathrm{obs}} \sim \mathcal N(r_i^{\mathrm{model}},\sigma_i^2),
$$

$$
\sigma_i=\sigma_{\mathrm{rate,abs}}+\sigma_{\mathrm{rate,rel}}r_i^{\mathrm{model}}.
$$

Supported error structures:

- `material`: one `sigma_rate_abs` and `sigma_rate_rel` pair per material.
- `shared`: one pair shared by all materials.

Individual fits require `material`. All-material fits may use either. LOMO requires `shared`, because an omitted material has no fitted material-specific noise parameters.

No log-rate IID, setup-intercept, MVN, or GP likelihood is part of the current active model path.

## Current finite-rate CO mechanism family

The current mechanism family uses:

- reversible finite-rate CO adsorption/desorption on Pd;
- exact detailed balance for CO desorption, `ln(k_-1) = ln(k1) - ln(K1)`;
- Pd-OH quasi-equilibrium;
- Ag-OH quasi-equilibrium when BF is active;
- analytic Pd CO steady state;
- BF, ER, and LH oxidation pathways in selectable subsets;
- stable log-sum-exp pathway addition.

Registered ordinary individual models:

```text
CO_LH
CO_ER
CO_BF
CO_ER_LH
CO_BF_LH
CO_BF_ER
CO_BF_ER_LH
```

Pure Pd individual fits are intentionally limited to:

```text
CO_LH
CO_ER
CO_ER_LH
```

because BF-specific parameters are structurally inactive on `Pd100`.

## All-material composition inference

All-material fits are handled by the same `fit_agpd_posterior.py` workflow using `--all-materials`; there is no separate composition-fitting script.

The principal all-material model is `CO_BF_ER_LH`. The actively used composition model is `linear_xAg`, referenced at

$$
x_{\mathrm{Ag,ref}}=0.5.
$$

For ordinary energetic parameters,

$$
p(x_{\mathrm{Ag}})=p_{0.5}+s_p(x_{\mathrm{Ag}}-0.5).
$$

For `beta_2_BF`, `beta_2_ER`, and `q`, the sampled `_xAg_slope` variable is a **relative slope factor**. It is scaled internally by

$$
2\min(p_{0.5},1-p_{0.5})
$$

before applying the linear composition shift. This keeps the effective parameter inside `[0,1]` over the full composition interval when `x_reference = 0.5`. Therefore the stored posterior slope for these bounded parameters is not directly the total physical change across composition.

The exact set and prior domains of active slopes are configuration-driven in `config/models/agpd_basic.yaml`.

## Shared prior source

All six materials now point to one canonical `CO_BF_ER_LH` prior profile in the model YAML.

For an individual model, the model registry projects that canonical parameter dictionary onto the fields required by the requested mechanism. Example: `Pd100 / CO_ER_LH` receives only

```text
deltaG1_0
deltaG4_0
beta_2_ER
Gact1_0
Gact2_ER_0
Gact2_LH_0
```

The all-material model receives the full parent profile plus any configured composition-slope priors.

This makes the base prior specification a single source of truth without adding inactive parameters to reduced individual mechanisms.

## Structural pathway masking in all-material fits

### Pd100

In the full all-material model, `Pd100` has `Ag_fraction = 0`. The BF rate contains the Ag surface-fraction factor, so its BF contribution is exactly zero. Pd100 therefore does not directly contribute BF-specific likelihood information.

For prediction-only pure-Pd states used by validation, the all-material wrapper explicitly evaluates the reduced Pd ER+LH mechanism while retaining compatibility with the training posterior parameter set.

### Ag10 no-BF variants

`CO_BF_ER_LH_Ag10_no_BF` uses the full parameterization but masks BF activity to zero on `Ag10Pd90`. Other alloy materials retain BF normally. BF-specific parameters are therefore learned from the remaining BF-active alloys, not from Ag10.

Equivalent masking is used by the capped/fitted-cap Ag10-no-BF variants.

## Coverage-cap variants

The registry contains exploratory fixed-cap and fitted-cap variants:

```text
CO_BF_ER_LH_capped
CO_BF_ER_LH_capped_Ag10_no_BF
CO_BF_ER_LH_fitted_caps_Ag10_no_BF
```

The current YAML explicitly marks coverage-cap approaches as exploratory / not preferred. The ordinary elementary-reaction models should remain the default scientific path unless a specific cap sensitivity analysis is intended.

## Posterior lifecycle and provenance

Posterior runs write `posterior.nc`, parameter summaries, metadata, and sampler-health information into deterministic run directories. Run metadata records data/config hashes, Git commit, sampler settings, fit scope, material/model identity, error structure, parameterization, and resolved parameterization specification.

`--resume` finalizes from an existing posterior checkpoint. `--overwrite` removes an existing run directory and resamples. They should not be used interchangeably.

A config-file hash warning after a harmless documentation/YAML refactor is intentionally not treated as proof of equivalence: the loader validates resolved run identity/parameterization but warns that other config values may have changed.

## Postprocessing implemented

`postprocess_agpd_posterior.py` supports individual and all-material runs and can generate:

- sampler diagnostics and sampling plots;
- posterior parameter summaries and prior/posterior contraction information;
- observation-level model and posterior-predictive rate distributions;
- residual products and material summaries;
- model-point coverages and pathway fractions;
- experimental alpha, OH-order, and adjacent CO-order comparisons;
- pointwise PSIS-LOO and Pareto-k diagnostics;
- analytic Normal LOO-PIT diagnostics;
- full/core/no-plot modes.

`postprocess_agpd_drc.py` now supports both individual and composition fits. Composition DRC perturbs effective composition-specific transition-state energies and supports half-step convergence checks.

`plot_agpd_composition_parameters.py` stores posterior effective-parameter trends versus Ag fraction.

## Predictive validation implemented

`validate_agpd_posterior.py` implements refit-based:

- **LOCO**: hold out one `(material, KOH, CO)` condition including all replicate curves.
- **LOMO**: hold out one material entirely.

LOCO may use `shared` or `material` error structures because the held-out material remains represented in training. LOMO requires `shared` errors unless a future hierarchical material-error model is introduced.

Validation produces held-out log predictive densities, PIT values, posterior/predictive rate summaries, residuals, posterior shifts relative to the full-data run, and figures. It also records whether the held-out composition is an extrapolation beyond the training composition range.

## Model comparison

`compare_agpd_models.py` compares completed runs with PSIS-LOO. It supports:

- individual models using `LABEL,MODEL` run specifications;
- all-material models using `LABEL,MODEL,PARAMETERIZATION,ERROR_STRUCTURE`.

Comparison weights are predictive combination weights, not posterior probabilities that a mechanism is true. Pareto-k failures must be inspected before interpreting ELPD differences.

## Known statistical limitations

The current likelihood treats observations as conditionally independent after the mechanistic mean function and heteroscedastic linear-rate error scale are specified. Experimental residuals can remain strongly structured along potential. Consequently:

- narrow posterior HDIs are not automatically calibrated physical uncertainties;
- observation-wise PSIS-LOO is not equivalent to predicting a new curve or new material;
- LOCO/LOMO are preferred for stronger generalization questions;
- better fit or LOO does not establish a microscopic mechanism.

A correlated residual model may be reconsidered later, but it is not currently active.

## Current development boundary

The codebase is now centered on the unified finite-rate CO model family and the linear-rate Normal likelihood. Future work should extend this architecture rather than revive retired likelihoods or duplicate individual/composition workflows.

High-value future directions include hierarchical/correlated residual structures, stronger mechanistic discrimination using pathway observables and DRC, and continued LOCO/LOMO evaluation of composition-dependent models.

## Physical and mathematical conventions

This file documents the physical and mathematical conventions used by the current AgPd basic-media model. The equations in `src/mkm/mechanisms/agpd_basic.py` and values in `config/models/agpd_basic.yaml` remain authoritative.

### Units and reference states

- Temperature: `293.15 K`.
- Electrode potential: `E_V_SHE`, volts versus SHE.
- Energies and activation free energies: eV.
- Rates: `s^-1` after Pd-ECSA normalization.
- Gas standard-state pressure: `1 bar`.
- Total gas pressure: `1 bar`.
- Electrolyte standard-state concentration: `1 M`.
- OH activity model: ideal molarity.

The model-point activities are

$$
a_{\mathrm{OH}}=\frac{C_{\mathrm{KOH}}}{1\;\mathrm{M}},
$$

and

$$
a_{\mathrm{CO}}=y_{\mathrm{CO}}\frac{P_{\mathrm{tot}}}{P^\circ}.
$$

At the current `P_tot = P^o = 1 bar`, `a_CO = y_CO`.

### Rate normalization

Raw current density is normalized on Pd ECSA. With two electrons per CO oxidation and a Pd site charge density of `210 uC cm^-2_Pd`,

$$
r=\frac{j}{2\times210}=\frac{j}{420}\;\mathrm{s^{-1}},
$$

when `j` is in `uA cm^-2_Pd`.

This is a Pd-ECSA-normalized turnover-like rate. Explicit Ag/Pd surface-fraction factors in the mechanism are ensemble/neighbor factors relative to that normalization; they are not a second ECSA correction.

### Surface composition

Current assumptions:

```text
surface_composition_equals_bulk: true
random_mixing: true
```

Configured surface fractions are:

| material | xAg | xPd |
| --- | ---: | ---: |
| Pd100 | 0.00 | 1.00 |
| Ag10Pd90 | 0.10 | 0.90 |
| Ag25Pd75 | 0.25 | 0.75 |
| Ag50Pd50 | 0.50 | 0.50 |
| Ag75Pd25 | 0.75 | 0.25 |
| Ag90Pd10 | 0.90 | 0.10 |

### Thermodynamics

Equilibrium constants use

$$
\ln K=-\frac{\Delta G}{k_BT}.
$$

Electrochemical reaction free energies use

$$
\Delta G(E)=\Delta G_0-nE.
$$

Current parameter meanings:

- `deltaG1_0`: standard free energy for CO adsorption on Pd.
- `deltaG4_0`: standard free energy for Pd-OH formation; one-electron potential dependence.
- `deltaG5_0`: standard free energy for Ag-OH formation; potential dependence scaled by `q`.

Thus

$$
\Delta G_4(E)=\Delta G_4^0-E,
$$

$$
\Delta G_5(E)=\Delta G_5^0-qE.
$$

### Transition-state kinetics

TST rate constants use

$$
\ln k=\ln\left(\frac{k_BT}{h}\right)-\frac{G^\ddagger}{k_BT}.
$$

Electrochemical activation free energies use

$$
G^\ddagger(E)=G_0^\ddagger-\beta nE.
$$

For ER,

$$
G^\ddagger_{\mathrm{ER}}(E)=G_{\mathrm{ER},0}^\ddagger-\beta_{\mathrm{ER}}E.
$$

For BF, Ag-OH formation consumes the fraction `q` of the overall electron-transfer dependence, so the BF oxidation barrier uses the remaining fraction:

$$
G^\ddagger_{\mathrm{BF}}(E)=G_{\mathrm{BF},0}^\ddagger-\beta_{\mathrm{BF}}(1-q)E.
$$

LH currently has no explicit potential dependence in its activation barrier:

$$
G^\ddagger_{\mathrm{LH}}=G_{\mathrm{LH},0}^\ddagger.
$$

### Reversible CO adsorption and detailed balance

Finite-rate CO models treat adsorption/desorption explicitly. The forward CO adsorption barrier is `Gact1_0` and

$$
\ln k_{-1}=\ln k_1-\ln K_1.
$$

Therefore the reverse rate constant follows exact detailed balance with the specified CO adsorption free energy.

The ordinary adsorption term is proportional to

$$
k_1 a_{\mathrm{CO}}\theta_{*,\mathrm{Pd}}.
$$

Exploratory capped models multiply adsorption by

$$
1-\frac{\theta_{\mathrm{CO}}}{\theta_{\mathrm{CO,max}}}.
$$

The cap models are retained for sensitivity work but are not the preferred default mechanism family.

### QEA coverages

Pd-OH remains in quasi-equilibrium with empty Pd sites:

$$
\theta_{\mathrm{OH,Pd}}=K_4 a_{\mathrm{OH}}\theta_{*,\mathrm{Pd}}.
$$

Ag-OH remains in quasi-equilibrium when BF is present:

$$
\theta_{\mathrm{OH,Ag}}=K_5 a_{\mathrm{OH}}\theta_{*,\mathrm{Ag}}.
$$

The ordinary site balances are

$$
\theta_{*,\mathrm{Pd}}+\theta_{\mathrm{OH,Pd}}+\theta_{\mathrm{CO}}=1,
$$

$$
\theta_{*,\mathrm{Ag}}+\theta_{\mathrm{OH,Ag}}=1.
$$

### CO steady state

CO coverage is determined from the steady-state balance between finite-rate adsorption/desorption and the active oxidation pathways. The solver remains analytic and PyTensor-compatible; no iterative numerical root solver is used inside HMC.

Without LH, the solution reduces to a linear site-balance form. With LH, the SSA becomes a quadratic that is solved analytically using numerically stable root expressions. The capped variant also remains quadratic.

### Pathway rate conventions

The pathway rates used by the current finite-rate models are proportional to:

BF:

$$
r_{\mathrm{BF}}=k_{2,\mathrm{BF}}\theta_{\mathrm{CO}}\theta_{\mathrm{OH,Ag}}x_{\mathrm{Ag}}.
$$

ER:

$$
r_{\mathrm{ER}}=k_{2,\mathrm{ER}}\theta_{\mathrm{CO}}a_{\mathrm{OH}}.
$$

LH:

$$
r_{\mathrm{LH}}=k_{2,\mathrm{LH}}\theta_{\mathrm{CO}}\theta_{\mathrm{OH,Pd}}x_{\mathrm{Pd}}.
$$

Active pathways are added in log space with log-sum-exp.

For `Pd100`, `xAg = 0`, so BF is exactly zero in the full all-material model. In `Ag10_no_BF` variants an explicit BF-activity mask makes the Ag10 BF term exactly zero.

### Composition parameterization

For ordinary parameters in `linear_xAg`,

$$
p(x)=p_{0.5}+s_p(x-0.5).
$$

The sampled `_xAg_slope` has units of the corresponding parameter per unit Ag fraction for unconstrained energy/barrier parameters.

For bounded parameters `beta_2_BF`, `beta_2_ER`, and `q`, the sampled value is a relative factor. Internally,

$$
s_{\mathrm{physical}}=s_{\mathrm{sampled}}\;2\min(p_{0.5},1-p_{0.5}),
$$

then

$$
p(x)=p_{0.5}+s_{\mathrm{physical}}(x-0.5).
$$

This is why the stored posterior `_xAg_slope` for these three parameters should be interpreted as a relative allowed change, not directly as the total physical change.

### Likelihood convention

The active observation model is

$$
r_i^{\mathrm{obs}}\sim\mathcal N(r_i^{\mathrm{model}},\sigma_i^2),
$$

with

$$
\sigma_i=\sigma_{\mathrm{rate,abs}}+\sigma_{\mathrm{rate,rel}}r_i^{\mathrm{model}}.
$$

Both error parameters are positive LogNormal RVs. `material` error structure indexes them by material; `shared` uses one pair across the fit.

The likelihood is conditionally independent across observations. Any residual correlation along potential is therefore model discrepancy not explicitly represented by the current error covariance.

### DRC convention

Transition-state degree of rate control is defined as

$$
X_{\mathrm{TS},j}=-k_BT\frac{\partial\ln r}{\partial G_j^\ddagger}.
$$

The implementation uses finite perturbations of transition-state free energies and can compare the default perturbation to a half-sized step as a convergence check. For all-material fits the perturbation is applied to the effective composition-specific transition-state energy.

## Model registry

This file documents the AgPd basic-media model registry implemented in `src/mkm/models/agpd_basic.py` and `src/mkm/mechanisms/agpd_basic.py`.

### Ordinary individual finite-rate CO models

| model | active pathways | physical parameters |
| --- | --- | --- |
| `CO_LH` | LH | `deltaG1_0`, `deltaG4_0`, `Gact1_0`, `Gact2_LH_0` |
| `CO_ER` | ER | `deltaG1_0`, `deltaG4_0`, `beta_2_ER`, `Gact1_0`, `Gact2_ER_0` |
| `CO_BF` | BF | `deltaG1_0`, `deltaG4_0`, `deltaG5_0`, `beta_2_BF`, `q`, `Gact1_0`, `Gact2_BF_0` |
| `CO_ER_LH` | ER + LH | `deltaG1_0`, `deltaG4_0`, `beta_2_ER`, `Gact1_0`, `Gact2_ER_0`, `Gact2_LH_0` |
| `CO_BF_LH` | BF + LH | `deltaG1_0`, `deltaG4_0`, `deltaG5_0`, `beta_2_BF`, `q`, `Gact1_0`, `Gact2_BF_0`, `Gact2_LH_0` |
| `CO_BF_ER` | BF + ER | `deltaG1_0`, `deltaG4_0`, `deltaG5_0`, `beta_2_BF`, `beta_2_ER`, `q`, `Gact1_0`, `Gact2_BF_0`, `Gact2_ER_0` |
| `CO_BF_ER_LH` | BF + ER + LH | all ten physical parameters |

All ordinary models use finite-rate reversible CO adsorption/desorption and the same analytic CO-SSA machinery. The pathway subset controls which oxidation terms and parameters are active.

### Material restrictions

All five Ag-containing alloys may use every ordinary individual model.

`Pd100` individual fits intentionally support only:

```text
CO_LH
CO_ER
CO_ER_LH
```

BF-containing individual models are rejected on pure Pd because BF-specific parameters are structurally inactive when `Ag_fraction = 0`.

### All-material models

Enabled all-material models:

```text
CO_BF_ER_LH
CO_BF_ER_LH_Ag10_no_BF
CO_BF_ER_LH_capped
CO_BF_ER_LH_capped_Ag10_no_BF
CO_BF_ER_LH_fitted_caps_Ag10_no_BF
```

The ordinary scientific default is `CO_BF_ER_LH` with a configured composition parameterization.

### `CO_BF_ER_LH_Ag10_no_BF`

Uses the same full parameter dataclass and composition parameterization as `CO_BF_ER_LH`, but sets BF activity to zero on `Ag10Pd90`.

Consequences:

- Ag10 contributes ER/LH/shared-chemistry information normally.
- Ag10 contributes no direct BF-pathway likelihood information.
- BF parameters remain global and are informed by the other BF-active alloys.
- Pd100 also has zero BF contribution because `Ag_fraction = 0`.

### Fixed-cap variants

`CO_BF_ER_LH_capped` and `CO_BF_ER_LH_capped_Ag10_no_BF` use material-specific fixed `theta_CO_max` values from the YAML. The cap modifies the CO adsorption factor while retaining the ordinary Pd site balance.

These are retained as exploratory sensitivity models and are currently marked in configuration as not preferred for the main mechanistic analysis.

### `CO_BF_ER_LH_fitted_caps_Ag10_no_BF`

Adds one fitted `theta_CO_max_<material>` RV per configured material and uses a calibration-specific parameter/slopes block. This is a sensitivity/calibration model, not the normal prior source for ordinary fits.

### Prior resolution

All six materials point to one canonical YAML prior profile:

```text
prior_profiles.<material>.CO_BF_ER_LH
```

For an ordinary individual model:

1. load the canonical `CO_BF_ER_LH` parent for that material;
2. inspect the requested model parameter dataclass;
3. project the parent dictionary onto exactly those required fields;
4. validate that no required parameter is missing.

Therefore a reduced mechanism never receives inactive parameters merely because the canonical parent is larger.

Because every material currently aliases the same parent profile, changing one base prior definition updates all ordinary materials consistently.

For all-material fitting, `--prior-material` selects the material whose canonical parent is used. With the current shared YAML aliases this choice resolves to the same base prior specification for every material, but the run metadata still records the requested prior material.

### Composition parameterizations

Parameterization availability is configuration-driven. All-material models resolve aliases back to the base `CO_BF_ER_LH` configuration when appropriate.

### `shared`

No composition slopes. One global physical parameter value is used at every composition.

### `linear_xAg`

Reference composition:

$$
x_{\mathrm{Ag,ref}}=0.5.
$$

For ordinary unbounded/energy-like parameters:

$$
p(x)=p_{0.5}+s_p(x-0.5).
$$

The current YAML may assign slopes to any subset of mechanism parameters. The exact current list and prior bounds should be read directly from `config/models/agpd_basic.yaml` rather than duplicated here.

For `beta_2_BF`, `beta_2_ER`, and `q`, the sampled slope is internally scaled by

$$
2\min(p_{0.5},1-p_{0.5})
$$

before the composition shift. These stored slope variables therefore represent relative allowed change.

### Pure-Pd behavior inside an all-material fit

The full all-material model may contain BF parameters globally, but a Pd100 point has `Ag_fraction = 0`. The BF apparent rate includes `log(Ag_fraction)`, which is `-inf`, so BF consumption and BF rate are exactly zero for Pd100.

Thus Pd100 has zero **direct** likelihood sensitivity to BF-specific parameters. It may still affect their marginal posterior indirectly through posterior correlation with parameters shared across pathways/materials, which is expected in joint Bayesian inference.

During prediction-only validation of a pure-Pd held-out state, the wrapper explicitly uses the reduced Pd ER+LH evaluator. This avoids requiring structurally inactive BF terms in the held-out prediction graph while preserving posterior parameter compatibility.

### Parameter names

Base physical parameters:

```text
deltaG1_0
deltaG4_0
deltaG5_0
beta_2_BF
beta_2_ER
q
Gact1_0
Gact2_BF_0
Gact2_ER_0
Gact2_LH_0
```

All-material composition slope parameters are named:

```text
<base_parameter>_xAg_slope
```

Error parameters:

```text
sigma_rate_abs
sigma_rate_rel
```

With `error_structure=material`, these have a material dimension. With `shared`, they are scalar RVs.

Fitted-cap parameters are named:

```text
theta_CO_max_<material>
```

### Model outputs

Mechanisms return `ln_rate` plus pointwise variables when applicable, including:

```text
theta_CO
theta_OH_Pd
theta_empty_Pd
theta_OH_Ag
theta_empty_Ag
ln_rate_BF
ln_rate_ER
ln_rate_LH
rate_fraction_BF
rate_fraction_ER
rate_fraction_LH
```

Only variables meaningful for the active pathway set are emitted.