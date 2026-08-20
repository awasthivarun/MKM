# CURRENT_STATE.md

## Project status

The rebuild has moved well beyond preprocessing. The current repository now contains a working AgPd basic-data pipeline, generic model-data and PyMC infrastructure, several AgPd microkinetic mechanisms, material/model prior profiles, and a prior-predictive checking framework.

**Current checkpoint:** 143 tests passing.

**Current Git checkpoint planned by the user:** `Prior predictive checks framework`

No posterior sampling has been performed in the rebuilt repository yet.

---

## Repository foundation

Completed:

- Clean rebuild branch and Conda environment.
- `src/` package layout with editable installation through `pip install -e .`.
- Immutable `data/raw/` policy.
- Dataset-specific raw-data documentation.
- `data/processed/`, `scripts/`, `config/`, and `tests/` structure.
- Pytest running from repository root.
- Current package includes preprocessing, model-data, observable-map, inference, mechanism, model-registry, diagnostics, and plotting utilities.

The repository is intentionally being rebuilt around actual observed conditions rather than rectangular concentration × pressure assumptions inherited from the old code.

---

## AgPd basic-data preprocessing

### Raw-data contract

Implemented and tested for:

- Materials:
  - `Pd100`
  - `Ag10Pd90`
  - `Ag25Pd75`
  - `Ag50Pd50`
  - `Ag75Pd25`
  - `Ag90Pd10`
- KOH concentrations: 0.25, 0.50, and 1.00 M.
- CO mole fractions: 0.001, 0.01, 0.10, and 1.00.
- Replicates: A/B/C.
- Potential reference: SHE.
- Raw current representation: `ln(j)`, with underlying `j` in µA/cm²_Pd.
- AgPd-specific expectation: 1000 acquired points per raw curve.

Raw ingestion preserves acquisition `point_index`, `ln_j_uA_cm2`, and reconstructed `j_uA_cm2`.

### Potential-grid processing

Implemented:

- Condition-by-condition replicate alignment.
- Increasing and decreasing monotonic sweeps handled explicitly.
- Non-monotonic sweeps rejected at the model-grid stage rather than silently sorted.
- No extrapolation beyond measured replicate domains.
- Interpolation performed in `ln(j)` using PCHIP.
- Canonical AgPd analysis grid:
  - spacing = 0.010 V
  - origin = 0.0 V
- `analysis_grid_index` is distinct from raw acquisition `point_index`.

The 10 mV grid is the fitting/observable grid. It reduces oversampling relative to the native ~0.4 mV spacing, but it is not assumed to make neighboring residuals statistically independent.

### Rate normalization

AgPd current density is normalized by Pd ECSA.

For CO oxidation:

\[
r = \frac{j}{2(210)}
  = \frac{j}{420}\;\mathrm{s^{-1}}
\]

when `j` is expressed in µA/cm²_Pd.

Interpretation:

- TOF is per Pd-ECSA-equivalent site.
- It is not a turnover rate per total Ag + Pd surface atom.
- Increasing Pd-normalized TOF with Ag content is therefore not by itself evidence for an electronic effect.

### Truncation

Current AgPd fitting-window rule:

- Low-potential threshold: \(r \ge 10^{-3}\;\mathrm{s^{-1}}\).
- All three replicates must satisfy the threshold.
- The threshold must remain satisfied through the condition-level mean-rate peak.
- No high-potential truncation.
- Post-peak data are retained even if the rate later falls below the low-potential threshold.

Current processed counts:

- Full analysis rows: 8856.
- Selected replicate rows: 8793.
- Selected summary points: 2931.
- Conditions truncated: 6.

### Experimental observables

Implemented:

\[
\alpha = \frac{RT}{F}\frac{d\ln r}{dE}
\]

using numerical differentiation on the processed potential grid.

Implemented reaction orders:

- `delta_OH`: slope of \(\ln r\) versus \(\ln C_{\mathrm{KOH}}\) using all three KOH concentrations where available.
- `delta_CO`: adjacent CO-mole-fraction slopes for:
  - 0.1 → 1%
  - 1 → 10%
  - 10 → 100%

Replicate-scale SD, not SEM, is currently propagated into experimental reaction-order uncertainty.

Open experimental-design issue:

- A/B/C replicates are likely paired across CO-pressure conditions, but this has not yet been confirmed from written experimental records.
- Current reaction-order uncertainty propagation therefore does not assume cross-condition covariance.

---

## Generic model-data architecture

Implemented in the rebuild:

### `model_data.py`

Creates three canonical tables:

- `conditions`
- `model_points`
- `observations`

Properties:

- Only actually observed condition combinations are represented.
- Stable contiguous `condition_id`, `model_point_id`, and `observation_id`.
- No full concentration × pressure Cartesian product is constructed.
- Replicates remain explicit observations.
- Condition-specific potential windows are supported.

### `model_inputs.py`

Creates PyMC/PyTensor-ready arrays for:

- material indexing
- condition indexing
- model-point indexing
- SHE potential
- log electrolyte concentration
- log CO mole fraction
- observation-to-model-point mapping
- replicate-resolved observed log rate

Also provides model-point-aligned inputs for mechanism evaluation.

### `observable_maps.py`

Generic fixed linear maps have been implemented for derived observables of model log rate:

- transfer coefficient
- electrolyte reaction order
- adjacent CO reaction order

The maps operate as

\[
q_k = \sum_j w_{kj}\ln r_j
\]

and therefore do not require rectangular condition arrays.

The map-derived AgPd observables have already been checked against preprocessing calculations on real selected data.

---

## Likelihood and residual diagnostics

Current baseline likelihood:

\[
\ln r_{i}^{\mathrm{obs}}
\sim
\mathcal N\left(
\ln r_{m(i)}^{\mathrm{model}},
\sigma_{\mathrm{material}(i)}
\right)
\]

with one learned log-rate residual scale per material.

Current prior:

\[
\sigma_{\ln r,\mathrm{material}}
\sim
\mathrm{LogNormal}
\left(
\ln 0.20,\;0.75
\right)
\]

This is intentionally a replaceable baseline likelihood.

Empirical replicate diagnostics showed:

- global pooled log-rate SD ≈ 0.210
- substantial material-to-material variation
- strong smooth replicate residual structure along potential
- replicate curves are therefore not well described as independent point noise

The current likelihood is consequently documented as an iid approximation. Potential autocorrelation and curve-level replicate effects remain candidates for later likelihood refinement.

---

## Generic mechanism and PyMC infrastructure

Implemented:

### Mechanism contract

A mechanism accepts model-point inputs and returns:

- one `ln_rate` prediction per model point
- optional pointwise quantities such as coverages, pathway rates, and pathway fractions

The PyMC model shell is chemistry-agnostic and:

- builds coordinates
- evaluates a mechanism
- registers model-point deterministics
- maps predictions to replicate observations
- attaches the log-rate likelihood

### Thermodynamic / kinetic primitives

AgPd basic mechanism utilities now include:

- thermal energy in eV
- Eyring/TST prefactor
- equilibrium constants from free energies
- electrochemical free-energy shifts
- electrochemical activation-energy shifts
- stable log-space site balances
- stable log-sum-exp pathway addition
- surface-composition prefactors

Current convention for an electrochemical free energy is:

\[
\Delta G(E) = \Delta G_0 - nE
\]

with energies in eV and potential in V.

For an electrochemical barrier:

\[
G^\ddagger(E)
=
G^\ddagger_0
-
\beta nE
\]

---

## AgPd model conventions

Current model configuration is separate from preprocessing configuration.

### Gas activity

Gas standard state:

\[
P^\circ = 1\;\mathrm{bar}
\]

Nominal cell total pressure:

\[
P_{\mathrm{tot}} = 1\;\mathrm{bar}
\]

Therefore:

\[
a_{\mathrm{CO}}
=
y_{\mathrm{CO}}
\frac{P_{\mathrm{tot}}}{P^\circ}
=
y_{\mathrm{CO}}
\]

for the current convention.

A local-pressure correction such as 0.85 bar would shift a one-CO free-energy term by only ~0.004 eV at 293.15 K. Because the actual cell/headspace pressure is not known to that accuracy, the nominal 1 bar convention is retained.

### Electrolyte activity

Current ideal-molarity approximation:

\[
a_{\mathrm{OH^-}}
=
\frac{C_{\mathrm{KOH}}}{1\;\mathrm{M}}
\]

No activity-coefficient correction is currently applied.

### Surface composition

Current mean-field assumptions:

- surface composition equals bulk composition
- random atomic mixing
- CO adsorbs only on Pd

Because experimental TOF is normalized per Pd ECSA site, the central reactive site is already conditioned to be Pd.

Therefore:

\[
r_{\mathrm{ER}}
=
r_{\mathrm{ER}}^{\mathrm{intrinsic}}
\]

\[
r_{\mathrm{LH}}
=
x_{\mathrm{Pd}}
r_{\mathrm{LH}}^{\mathrm{intrinsic}}
\]

\[
r_{\mathrm{BF}}
=
x_{\mathrm{Ag}}
r_{\mathrm{BF}}^{\mathrm{intrinsic}}
\]

The composition factors represent the mean-field probability that the required neighboring atom is Pd or Ag.

---

## AgPd mechanisms implemented

The following kernels are implemented and tested.

### QEA surface-state models

Pd competitive adsorption:

\[
\theta_* =
\frac{1}
{1 + K_{\mathrm{CO}}a_{\mathrm{CO}}
+ K_{\mathrm{OH,Pd}}a_{\mathrm{OH}}}
\]

with corresponding \(\theta_{\mathrm{CO}}\) and \(\theta_{\mathrm{OH,Pd}}\).

Ag OH adsorption:

\[
\theta_\# =
\frac{1}
{1 + K_{\mathrm{OH,Ag}}a_{\mathrm{OH}}}
\]

with corresponding \(\theta_{\mathrm{OH,Ag}}\).

### Pathways

Implemented:

- BF
- LH
- ER
- BF + LH
- BF + ER + LH

The partial-charge-transfer BF construction is:

\[
\Delta G_5(E)
=
\Delta G_{5,0} - qE
\]

and

\[
G^\ddagger_{\mathrm{BF}}(E)
=
G^\ddagger_{\mathrm{BF},0}
-
\beta_{\mathrm{BF}}(1-q)E
\]

so the OH adsorption + BF oxidation sequence transfers \(q + (1-q)=1\) electron.

### CO adsorption under SSA

Implemented for the `CO_BF_ER_LH` model.

CO adsorption/desorption is treated kinetically while Pd-OH and Ag-OH remain in QEA.

The Pd site balance and CO steady-state equation reduce analytically to a quadratic in the empty-Pd-site coverage. The implemented form is scaled by the total first-order CO-consumption coefficient before solving, improving numerical conditioning.

The physical root is evaluated analytically and the resulting coverages are verified to:

- remain bounded
- satisfy the Pd site balance
- satisfy the CO steady-state balance
- recover the CO-QEA limit when CO consumption becomes negligible

No iterative nonlinear solver is required.

---

## Model registry and priors

Current registered Ag10Pd90 models of interest:

- `BF`
- `BF_LH`
- `CO_BF_ER_LH`

Model equations and prior definitions are separate.

The generic prior builder currently supports:

- Normal
- Uniform
- Truncated Normal

The Ag10Pd90 prior profiles were taken from the legacy `base_error` analysis and are explicitly labeled:

`legacy_tuned_for_multimodality`

These priors were historically tuned to suppress multimodal solutions and are not treated as mechanistically neutral physical priors.

The current legacy-style mechanism builder intentionally enforces one material per fit. A future joint-composition model must introduce material-specific or composition-dependent parameters explicitly rather than accidentally sharing one scalar parameter set across all alloys.

---

## Prior-predictive framework

Implemented and tested.

Current script:

`scripts/check_agpd_prior_predictive.py`

Current real-data prior-predictive check:

- material: `Ag10Pd90`
- models:
  - BF
  - BF_LH
  - CO_BF_ER_LH
- draws: 2000
- real selected Ag10Pd90 model-point design
- persistent NetCDF output plus tabular summaries

### Current prior-predictive findings

These findings describe the priors, not evidence for a mechanism.

#### BF

Prior-median latent log-rate range:

\[
-9.813 \rightarrow -1.700
\]

QEA Pd surface is strongly biased toward CO saturation at the prior median:

\[
\theta_{\mathrm{CO}}^{50\%}
\approx
0.885 \rightarrow 1
\]

while median empty-Pd coverage is very small.

#### BF + LH

Prior-median latent log-rate range:

\[
-8.750 \rightarrow 0.704
\]

The same QEA CO-saturation tendency is present.

Prior-median pathway fractions:

\[
f_{\mathrm{BF}}
\approx
0.967 \rightarrow 1
\]

\[
f_{\mathrm{LH}}
\approx
1.9\times10^{-7}
\rightarrow
0.0326
\]

Thus the typical prior state is effectively BF-dominated, although the broad prior still permits LH-dominated states.

#### CO-SSA + BF + ER + LH

Prior-median latent log-rate range:

\[
-11.362 \rightarrow -2.466
\]

The CO-SSA surface explores much broader Pd states:

- median \(\theta_{\mathrm{CO}}\): approximately \(3.3\times10^{-8}\) to 0.982
- median \(\theta_{\mathrm{OH,Pd}}\): approximately \(9\times10^{-8}\) to 0.997
- median \(\theta_*\): approximately \(7.6\times10^{-4}\) to 0.886

Prior-median pathway fractions:

\[
f_{\mathrm{ER}}
\approx
0.337 \rightarrow 0.926
\]

\[
f_{\mathrm{BF}}
\approx
4.1\times10^{-4}
\rightarrow
3.2\times10^{-3}
\]

\[
f_{\mathrm{LH}}
\approx
6.2\times10^{-6}
\rightarrow
0.019
\]

Therefore the typical prior state of this model is strongly ER-dominated.

The 95% envelopes for pathway fractions remain broad enough to include alternative pathway dominance. The priors are therefore not prohibitive, but they are strongly non-neutral.

### Current decision on priors

Do not redesign these priors before the first rebuilt posterior fits.

Reason:

- they provide a controlled bridge to the legacy analyses
- changing the model implementation, likelihood, and priors simultaneously would make differences from the old results difficult to diagnose

Required later:

- prior-to-posterior contraction
- boundary-accumulation checks
- targeted prior-sensitivity analyses
- physical plausibility checks before mechanistic interpretation

---

## Known unresolved issues

1. **Replicate correlation across potential**

   Smooth replicate residual trajectories show that the current iid pointwise likelihood is only an approximation.

2. **Possible pairing of replicate labels across experimental conditions**

   A/B/C may be experimentally paired across CO pressures, but this still requires confirmation before introducing covariance into reaction-order uncertainty or the likelihood.

3. **Legacy prior influence**

   The current Ag10Pd90 priors were tuned to remove multimodality and strongly shape prior coverages/pathway allocations.

4. **Only Ag10Pd90 prior profiles are currently rebuilt**

   Other alloy compositions had different tuned priors and, in some cases, different preferred model complexity in the old notebooks. Their model profiles must be extracted from the corresponding source notebooks rather than inferred from Ag10Pd90.

5. **No rebuilt posterior inference has yet been run**

   No mechanistic conclusion should currently be drawn from fit quality, pathway fractions, or prior-predictive behavior.

6. **DRC infrastructure has not yet been rebuilt**

   Future DRC calculations should perturb thermodynamically meaningful quantities consistently rather than independently perturbing forward/reverse rate constants.

7. **PtRu preprocessing/modeling remains deferred**

   The architecture is being designed to support PtRu acid/basic later, but current implementation work is focused on AgPd basic data.

---

## Immediate repository checkpoint and code-condensation pass

The user is currently pausing scientific development to:

1. Commit/push the current repository state with the checkpoint message:

   `Prior predictive checks framework`

2. Run a local code-editing bot across the repository to reduce unnecessary vertical formatting and make the code more compact/readable.

This refactor is intended to be **behavior-preserving**:

- do not change equations
- do not change parameter names
- do not change public interfaces
- do not change data contracts
- do not change model/prior semantics
- do not change test meaning

After the condensation pass, the first validation step should be:

```text
pytest tests -v
```

Expected baseline before refactoring:

**143 passed**

A prior-predictive script rerun is also advisable after a broad automated refactor, especially if the bot changes anything beyond whitespace/line wrapping.

---

## Plan ahead

### Next scientific step

Build a generic draw-based derived-observable layer using the existing `LinearObservableMap` infrastructure.

The goal is to apply the same fixed maps to prior and posterior draws of model log rate and obtain distributions of:

\[
\alpha
\]

\[
\delta_{\mathrm{OH}}
\]

\[
\delta_{\mathrm{CO}}
\]

without introducing AgPd-specific rectangular indexing.

Then rerun prior-predictive checks for Ag10Pd90 and determine whether the legacy priors place reasonable probability mass over the experimentally diagnostic transfer coefficients and reaction orders.

### First posterior fits

After prior-predictive observable checks:

- fit Ag10Pd90 BF
- fit Ag10Pd90 BF + LH
- fit Ag10Pd90 CO-SSA + BF + ER + LH

For each fit, inspect separately:

- sampler diagnostics
- parameter identifiability
- prior-to-posterior contraction
- parameter-boundary accumulation
- coverages/site balances
- pathway fractions
- posterior predictive behavior
- residual structure
- experimental/model \(\alpha\)
- experimental/model \(\delta_{\mathrm{OH}}\)
- experimental/model \(\delta_{\mathrm{CO}}\)

Statistical improvement is not sufficient evidence for a mechanism.

### After the first Ag10Pd90 fits

- Rebuild thermodynamically consistent DRC calculations.
- Compare models on identical observation sets.
- Inspect LOO/Pareto-k only after sampler and physical diagnostics are acceptable.
- Perform prior-sensitivity fits for parameters whose posterior interpretation depends strongly on legacy truncation.
- Revisit the iid likelihood if residual correlation remains important.
- Extract material-specific prior/model profiles from the other AgPd notebooks.
- Repeat the validated workflow across the remaining AgPd compositions.
- Only then move toward cross-composition mechanistic interpretation or joint composition-dependent models.

---

## Scientific interpretation status

### Mathematically established

- AgPd preprocessing and normalization pipeline.
- Nonrectangular model-data indexing.
- Linear observable-map construction.
- QEA site balances.
- Stable parallel-pathway log-rate construction.
- Analytic CO-SSA solution and its limiting/steady-state checks.
- Replicate-resolved log-rate likelihood implementation.
- Prior-predictive sampling framework.

### Supported current diagnostic statements

- Legacy QEA priors strongly favor CO-covered Pd surfaces.
- The Ag10Pd90 BF + LH prior typically favors BF before observing data.
- The Ag10Pd90 CO-SSA + BF + ER + LH prior typically favors ER before observing data.
- Experimental replicate residuals show smooth structure along potential inconsistent with a strictly independent point-noise interpretation.

### Not yet established

- Which Ag10Pd90 mechanism is preferred under the rebuilt likelihood.
- Whether LH, ER, BF, or CO adsorption has significant posterior DRC.
- Whether prior-tuned multimodality has been eliminated without distorting mechanistic inference.
- Whether the rebuilt results reproduce the legacy model ranking.
- Whether any pathway assignment generalizes across AgPd compositions.
- Whether observed composition trends are electronic, bifunctional, structural, or some combination.

No mechanistic conclusion should be promoted beyond these evidence levels until posterior inference and sensitivity analyses are complete.
