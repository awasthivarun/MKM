# REPO_MAP

Architecture and ownership guide for the MKM rebuild.

## End-to-end AgPd flow

```text
raw AgPd Excel workbooks
        |
        v
config/preprocessing/agpd_basic.yaml
        |
        v
scripts/process_agpd_basic.py
        |
        v
src/mkm/preprocessing/*
        |
        v
processed parquet products
        |
        v
src/mkm/model_data.py
  conditions / model_points / observations
        |
        v
src/mkm/model_inputs.py
        |
        +-----------------------+
        |                       |
        v                       v
src/mkm/models/agpd_registry.py + agpd_basic.py    src/mkm/mechanisms/agpd_basic/
        |                       |
        +-----------+-----------+
                    |
                    v
          src/mkm/inference/model.py
                    |
                    v
       src/mkm/inference/likelihoods.py
                    |
                    v
        scripts/fit_agpd_posterior.py
                    |
                    v
               posterior.nc
                    |
        +-----------+-----------+----------------+
        |                       |                |
        v                       v                v
postprocess_agpd_posterior.py  postprocess_agpd_drc.py  plot_agpd_composition_parameters.py
        |
        v
src/mkm/postprocessing/*

completed all-material posterior
        |
        +--> compare_agpd_models.py
        |
        +--> validate_agpd_posterior.py --> LOCO / LOMO refits + held-out prediction
```

The old split between individual fitting and separate composition-fitting scripts no longer exists. `fit_agpd_posterior.py`, `postprocess_agpd_posterior.py`, and `postprocess_agpd_drc.py` each handle both scopes through `--all-materials`.

## Unified PtRu preprocessing and inference flow

PtRu experimental preprocessing covers Pt100, Pt66Ru33, and Pt50Ru50 in
acidic/basic electrolyte, with Todd and Em data handled separately.
Microkinetic inference supports individual Pt100, Pt66Ru33, and Pt50Ru50 fits, plus
all-material fits within an acidic or basic medium. The media are never combined.

```text
Todd / Em raw Excel workbooks
             |
             v
config/preprocessing/ptru_todd.yaml
             |
             v
process_ptru_todd.py / process_ptru_em.py
             |
             v
ptru_todd.py / ptru_em.py
             |
             v
Standardized and full analysis Parquets
             |
             +-------------------------+
             |                         |
             v                         v
plot_ptru_todd.py       process_ptru_observables.py
             |                         |
             v                         v
ptru_plotting.py        ptru_observables.py
             |                         |
             |                         v
             |                Shared observables.py
             |                         |
             +------------+------------+
                          |
                          v
             Experimental figures and
             kinetic-observable Parquets
```

Selected Todd PtRu observations enter the common fitting stack; Pt100 and alloys
use the same evaluator, registry, likelihood, and posterior code:

```text
Todd selected observations (acidic or basic, independently)
             |
             v
ptru_fit.py / models/ptru.py / mechanisms/ptru.py
             |
             v
scripts/fit_ptru_posterior.py --> posterior.nc
             |
             +--> scripts/postprocess_ptru_posterior.py (includes optional DRC)
             +--> scripts/postprocess_ptru_drc.py (standalone DRC)
             +--> scripts/compare_ptru_models.py (compatible fitted models)
             +--> scripts/validate_ptru_posterior.py (individual LOCO)
             +--> scripts/validate_ptru_lomo.py (pooled LOMO, shared errors)
             +--> scripts/plot_ptru_composition_parameters.py (pooled parameter trends)
```

### Configuration

`config/preprocessing/ptru_todd.yaml` owns the experimental
preprocessing conventions for both Todd and Em:

- temperature (293.15 K);
- primary Pt/PtRu materials;
- Todd electrolyte and CO concentration grids;
- ideal electrolyte-activity convention and pKw;
- Pt CO-stripping ECSA rate-normalization constants;
- common SHE analysis-grid spacing and origin.

Em's additional CO concentrations and sample preparations are
identified from the source-specific data.

PtRu mechanism and prior settings are defined in `config/models/ptru.yaml`
and `config/priors/ptru.yaml`. Pure Pt supports `CO_LH`, `CO_ER`, and `CO_ER_LH`.
PtRu alloys additionally support BF pathway combinations. All-material parameterizations
are `shared`, `independent`, and `linear_xRu`; acid and base remain separate.

### Module ownership

`src/mkm/preprocessing/ptru_todd.py`

- Todd workbook parsing and validation;
- raw-data standardization;
- SHE conversion and rate normalization;
- PCHIP log-rate interpolation onto the analysis grid.

`src/mkm/preprocessing/ptru_em.py`

- Em Pt/C workbook parsing;
- sample-preparation and replicate identification;
- source-specific data standardization.

Em uses the shared Todd analysis-grid construction function.

`src/mkm/preprocessing/ptru_observables.py`

- replicate-resolved transfer coefficients;
- adjacent-CO-pressure reaction orders;
- full-range H+ and OH- reaction orders;
- PtRu-specific grouping and replicate-uncertainty conventions.

Reusable mathematical operations remain in
`src/mkm/preprocessing/observables.py`.

`src/mkm/preprocessing/ptru_plotting.py`

- experimental rate figures;
- transfer-coefficient figures;
- CO reaction-order figures;
- electrolyte reaction-order figures;
- PtRu material colors, panel layouts, and tick formatting.

Shared plotting utilities remain in
`src/mkm/preprocessing/plotting.py`.

### Canonical PtRu scripts

```text
scripts/process_ptru_todd.py
scripts/process_ptru_em.py
scripts/plot_ptru_todd.py
scripts/process_ptru_observables.py
```

The consolidated observable script replaces:

```text
scripts/process_ptru_alpha.py
scripts/process_ptru_co_orders.py
scripts/process_ptru_ion_orders.py
```

The replaced entry points are not present in this repository snapshot.

### Scientific boundaries

The two catalyst families intentionally differ in preprocessing:

- AgPd CO-order calculations use paired replicate information.
- PtRu CO-order calculations use independent condition-mean rates.
- AgPd electrolyte orders use three KOH concentrations.
- PtRu electrolyte orders use four acid/base concentrations.
- PtRu retains full data and separately generates selected analysis datasets.
- AgPd has an established selected/truncated analysis dataset.
- Em Pt/C preparations are processed separately from Todd.

Experimental alpha and reaction orders are derived from rate data.
They are not independent measurements for likelihood construction.

## Configuration

### `config/preprocessing/agpd_basic.yaml`

Owns experimental/preprocessing facts:

- materials;
- KOH and CO grids;
- replicate labels and pairing;
- potential reference and acquisition assumptions;
- analysis-grid spacing;
- rate normalization;
- truncation rules.

### `config/models/agpd_basic.yaml`

Owns model/inference assumptions:

- temperature and activity reference states;
- material surface compositions;
- random-mixing/bulk-equals-surface assumptions;
- exploratory coverage-cap settings;
- composition parameterizations and slope priors;
- active linear-rate likelihood priors;
- shared canonical physical-parameter prior profile;
- fitted-cap calibration specification.

These two YAMLs serve different purposes and should remain separate.

## Scientific implementation

### `src/mkm/preprocessing/`

Owns raw workbook ingestion, validation, interpolation, rate normalization, truncation, and experimental observable construction.

### `src/mkm/model_data.py`

Builds the canonical non-rectangular tables:

```text
conditions
model_points
observations
```

Only observed combinations are represented.

### `src/mkm/model_inputs.py`

Converts model tables into contiguous NumPy index arrays and model-point mechanism inputs. Replicate observations map back to shared model-point means.

### `src/mkm/mechanisms/agpd_basic/`

Owns the chemical mathematics:

- activities and surface-composition state;
- equilibrium constants and TST rate constants;
- electrochemical free-energy/barrier conventions through shared `mkm.mechanisms.electrochem`;
- Pd-OH and Ag-OH QEA coverages;
- reversible finite-rate CO adsorption/desorption;
- analytic CO steady-state solver;
- BF, ER, LH pathway rates;
- stable pathway log-sum-exp;
- coverage-cap variants;
- Ag10 BF-activity masking;
- mechanism parameter/result dataclasses.

The `surface.py` helper `rate_fraction_from_log_rates` returns zero reported
pathway share when the total rate is exactly zero (`ln_rate = -inf`);
it does not change rate calculations. It is used by QEA and finite-CO
evaluators. BF-only dynamic evaluation on Pd100 therefore has zero rate
and a zero reported BF share, even though some BF/Pd100 fits remain
restricted by `workflows/agpd_fit.py`.

This package contains AgPd-specific chemistry, not PyMC prior choices or filesystem behavior.

### `src/mkm/mechanisms/electrochem.py`

Neutral shared PyTensor-compatible electrochemical free-energy, equilibrium,
transition-state-rate, and temperature primitives used by both AgPd and Pt100.
`src/mkm/mechanisms/agpd_basic/electrochem.py` re-exports these functions to
preserve older imports; it no longer owns their implementations.

### `src/mkm/mechanisms/ptru.py` and `src/mkm/models/ptru.py`

Unified finite-rate Pt CO SSA, Pt/Ru OH QEA, and BF/ER/LH pathway subsets in acid or base.
Ru OH adsorption is one-electron; no fitted `q` or `beta_2_BF` exists. Pure Pt has no BF
pathway. `models/ptru.py` registers single- and all-material priors and parameterizations.

### `src/mkm/mechanisms/pd_basic.py`

Owns the reduced pure-Pd ER+LH evaluator used for prediction-only pure-Pd states in all-material validation.

### `src/mkm/models/agpd_registry.py`

Declarative source of truth for AgPd model variants: active pathways, fit scopes, fixed parameters, named prior variants, coverage-cap mode, and material-specific pathway masks.

### `src/mkm/models/agpd_evaluator_settings.py`

Single resolver for mechanism-specific evaluation configuration passed by
AgPd fitting, posterior provenance, and DRC calculations. The registry and
physical mechanism implementations remain separate.

### `src/mkm/models/composition_bounds.py`

Shared mathematical helper for bounded relative slopes in AgPd `linear_xAg`
and PtRu `linear_xRu` models. Each caller supplies its own composition domain,
parameter bounds, and symbolic/array minimum operation.

### `src/mkm/models/agpd_basic.py`

Owns fit-ready mechanism registration:

- model name -> parameter dataclass/evaluator;
- individual/all-material model availability;
- prior-profile projection;
- model-config aliases for special variants;
- composition parameterization resolution;
- effective parameter construction versus `xAg`;
- relative-slope handling for bounded `beta/q` parameters;
- all-material pure-Pd prediction routing;
- fitted-cap assembly.

This is the main bridge between chemistry and inference configuration.

## Inference

### `src/mkm/inference/priors.py`

Builds named PyMC RVs from YAML prior specifications.

### `src/mkm/inference/likelihoods.py`

Owns the active likelihood:

$$
\sigma=\sigma_{\mathrm{rate,abs}}+\sigma_{\mathrm{rate,rel}}r_{\mathrm{model}}.
$$

Supports `shared` and `material` error structures.

### `src/mkm/inference/model.py`

Generic PyMC assembly: creates coordinates, evaluates the mechanism, stores deterministics, and adds the selected likelihood.

### `src/mkm/inference/posterior.py`

Sampling/reconstruction utilities used by the posterior lifecycle, including chain/draw coordinate-alignment checks.

## Workflow orchestration

### `src/mkm/workflows/agpd_postprocess/`

Postprocessing orchestration split into `workflow.py`, `products.py`, `plots.py`, and `status.py`; the CLI script is intentionally thin.

### `src/mkm/workflows/agpd_basic.py`

Loads AgPd configs/data and creates model tables/inputs.

### `src/mkm/workflows/agpd_fit.py`

Owns fit identity and assembly:

- `AgPdFitSpecification`;
- individual vs all-material validation;
- pure-Pd individual-model restrictions;
- allowed parameterizations/error structures;
- fit parameter/error specs;
- all-material mechanism construction;
- output directory resolution.

### `src/mkm/workflows/agpd_posterior.py`

Loads completed runs, reconstructs deterministics as needed, and validates run metadata/provenance against the requested fit identity.

### `src/mkm/workflows/posterior_lifecycle.py`

Owns checkpoint/sample/finalize behavior, including `sampling` / `sampled_free_variables` / `complete` states, `--resume`, `--overwrite`, metadata, sampler health, and checkpoint recovery.

### `src/mkm/workflows/agpd_validation.py`

Owns AgPd LOCO/LOMO split logic and held-out prediction:

- split selected data;
- enforce error-structure support;
- construct prediction-only mechanism graphs;
- compute held-out log predictive density and PIT;
- summarize held-out model/predictive distributions.

### `src/mkm/workflows/ptru_fit.py` / `ptru_posterior.py` / `ptru_validation.py`

The same PtRu fit builder supports individual material and pooled within-medium inference,
while LOCO uses an individual-material training subset and the same fitted model. Both
acidic and basic datasets use their own input identity and result directories.
`ptru_posterior.py` validates source signatures scoped to the selected mechanism,
resolved model/prior settings, and data identity. The whole-YAML hashes are recorded for
auditing but are not enforced for unrelated schema-2 configuration edits. Historical
checkpoints created under dedicated Pt100 or alloy modules are not automatically migrated.
`ptru_validation.py` also supports pooled LOMO with `shared` or `linear_xRu` physical
parameterization and shared errors; pooled LOCO remains unavailable.

### `src/mkm/workflows/ptru_postprocess.py` and internal helpers

`ptru_postprocess.py` remains the public orchestration/API module for individual
and pooled PtRu postprocessing. Responsibilities are divided into:

- `_ptru_postprocess_products.py`: experimental observable loading and posterior comparisons;
- `_ptru_postprocess_plots.py`: figure orchestration using shared plotting functions;
- `_ptru_postprocess_drc.py`: PtRu DRC output generation and status recording.

The public wrapper preserves established entry points (including plotting
callbacks used by existing regression tests). Per-material coverages, rates,
rate fractions, kinetic observables, and DRC remain supported.

## Postprocessing

### `src/mkm/postprocessing/diagnostics.py`

Parameter summaries, sample flattening, physical checks.

### `src/mkm/postprocessing/drc.py` and `drc_status.py`

AgPd/PtRu transition-state DRC finite differences, including pooled effective
barriers and half-step comparisons. `drc_status.py` records which DRC artifacts
were recomputed or retained from a previous invocation and protects existing
DRC plots when ordinary figures are regenerated. See
`tests/postprocessing/test_pooled_drc_regression.py` and `test_drc_status.py`.

### `sampling.py`

Sampler parameter names and sampler-health plotting helpers.

### `predictions.py`

Observation-level posterior mechanism and posterior-predictive diagnostics.

### `residuals.py`

Residual-curve structure and shared-vs-replicate summaries.

### `observables.py` / `observable_comparison.py`

Posterior linear observables and comparison to experimental alpha/OH-order/CO-order products.

### `second_order.py`

Potential derivatives of the first-order kinetic observables and the second-order kinetic-difference products used in posterior/experimental comparison.

### `loo.py`

Single-model PSIS-LOO and pointwise Pareto-k diagnostics.

### `calibration.py`

Normal-model LOO-PIT calculations.

### `model_comparison.py`

Multi-model ELPD comparison and pointwise differences.

### `drc.py`

Shared transition-state DRC and finite-step convergence checks for AgPd and PtRu individual/all-material models in their supported media.

### `composition_parameters.py` / `ptru_composition.py`

Effective physical-parameter trends versus Ag or Ru composition and the corresponding composition-overview figures. The PtRu implementation handles `shared`, `independent`, and `linear_xRu` pooled fits.

### `materials.py`

Material-level aggregation of residual/LOO/observation diagnostics.

### `validation.py`

Validation posterior-shift summaries and held-out validation plots.

### `plotting/`

Shared plotting package for posterior, DRC, and validation products, including
`observables.py` (electrolyte-configurable overlays) and `loo.py` (common LOO
figures). The AgPd wrappers retain their existing output conventions; PtRu
reuses the shared functions with HClO4/H+ or KOH/OH- labels.

## Test ownership and execution

- `tests/test_agpd_zero_rate_fractions.py` verifies zero total rate and
  defined zero reporting fractions for BF-only Pd100 dynamic evaluators.
- `tests/postprocessing/test_agpd_dynamic_drc.py` checks DRC sum rules,
  half-step stability, and uniform transition-state barrier shifts
  for selected reversible and interacting AgPd mechanisms (`slow`).
- `tests/test_agpd_lateral_branch_validation.py` explicitly requires both
  Frumkin branches to satisfy the terminal balance residual tolerance.
- `tests/test_agpd_lateral_adaptive_solver.py` covers difficult draws, early
  stopping, branch rejection, and symbolic gradients.
- `tests/test_agpd_reversible_subsets.py` retains its physics assertions but
  compiles related observables together to avoid repeated compilation.
- `tests/preprocessing/test_agpd_basic.py` shares expensive read-only data
  preparation while giving tests independent working copies.
- `tests/workflows/test_agpd_grid_runner.py` uses mocked subprocesses to check
  fit-grid retries and failures without running posterior inference.
- `tests/workflows/test_ptru_postprocessing_smoke.py` exercises the reorganized
  PtRu postprocessing across both electrolyte media.

The full suite is `pytest tests -q`; it remains the regression baseline.
`pyproject.toml` registers `slow`, `sampling`, `plotting`, and `integration`
markers, used by selected tests. The filtered command below excludes these
checks but is **not** comprehensive:

```powershell
pytest tests -q -m "not slow and not sampling and not plotting and not integration"
```

On 19 November 2026, the full suite passed 790 tests in about 3 minutes
before 13 additional zero-rate and DRC test cases were added;
the filtered subset required about 2 minutes 30 seconds. Use
`pytest tests -q --durations=30` to investigate test execution cost.

## Canonical AgPd scripts

```text
process_agpd_basic.py
plot_agpd_basic.py
check_agpd_prior_predictive.py
fit_agpd_posterior.py
postprocess_agpd_posterior.py
postprocess_agpd_drc.py
plot_agpd_composition_parameters.py
compare_agpd_models.py
run_agpd_individual_grid.py
validate_agpd_posterior.py
```

See `scripts/README.md` for CLI examples.

Preprocessing commands write source/raw-data/configuration/output lineage manifests alongside processed products. Posterior postprocessing maintains a per-invocation manifest and stage-status table, and clears only known stage-owned stale tables.

## AgPd result hierarchy

Individual posterior:

```text
results/AgPd_COOx_basic/posterior/individual/<material>/<model>/
```

All-material posterior:

```text
results/AgPd_COOx_basic/posterior/all_materials/<parameterization>/<error_structure>/<model>/
```

Inside a fit directory, products are organized primarily as:

```text
posterior.nc
posterior_parameters.csv
run_metadata.yaml
postprocessing_manifest.yaml
sampling_health.*
tables/  (including postprocessing_status.csv)
figures/
drc/
composition/
validation/
```

Exact filenames are owned by the producing scripts and may evolve while the fit-directory identity remains stable.

Individual model comparison:

```text
results/AgPd_COOx_basic/posterior/individual/<material>/model_comparison/<comparison_name>/
```

All-material comparison:

```text
results/AgPd_COOx_basic/posterior/all_materials/model_comparison/<comparison_name>/
```

LOCO/LOMO validation is nested under the corresponding full all-material fit directory:

```text
.../<model>/validation/loco/<material>/KOH_<...>_CO_<...>/
.../<model>/validation/lomo/<material>/
```

## Where do I change...?

### PtRu preprocessing

- Todd raw ingestion: `src/mkm/preprocessing/ptru_todd.py`
- Em raw ingestion: `src/mkm/preprocessing/ptru_em.py`
- PtRu preprocessing configuration: `config/preprocessing/ptru_todd.yaml`
- PtRu alpha and orders: `src/mkm/preprocessing/ptru_observables.py`
- PtRu experimental plots: `src/mkm/preprocessing/ptru_plotting.py`
- Shared observable mathematics: `src/mkm/preprocessing/observables.py`
- Shared plot styling: `src/mkm/preprocessing/plotting.py`
- PtRu CLI orchestration: `scripts/process_ptru_observables.py`

### AgPd and shared inference implementation

- raw workbook parsing: `src/mkm/preprocessing/agpd_basic.py`
- preprocessing facts: `config/preprocessing/agpd_basic.yaml`
- interpolation/truncation: `src/mkm/preprocessing/`
- experimental alpha/orders: `src/mkm/preprocessing/observables.py`
- model data indexing: `src/mkm/model_data.py`, `src/mkm/model_inputs.py`
- a mechanism equation: `src/mkm/mechanisms/agpd_basic/`
- pathway parameter dataclasses: `src/mkm/mechanisms/agpd_basic/finite_co.py`
- model registry / special variants: `src/mkm/models/agpd_registry.py`
- base/slope/likelihood config: `config/models/agpd_basic.yaml`; canonical priors/variants: `config/priors/agpd_basic.yaml`; exploratory cap variants: `src/mkm/models/agpd_registry.py`; additional experimental cap settings: `config/models/experimental/agpd_caps.yaml`
- PyMC prior factory: `src/mkm/inference/priors.py`
- likelihood equation: `src/mkm/inference/likelihoods.py`
- fit-scope rules: `src/mkm/workflows/agpd_fit.py`
- posterior provenance/load rules: `src/mkm/workflows/agpd_posterior.py`
- sampling lifecycle: `src/mkm/workflows/posterior_lifecycle.py`
- LOCO/LOMO splitting/prediction: `src/mkm/workflows/agpd_validation.py`
- posterior diagnostics/plots: `src/mkm/postprocessing/`
- canonical CLI behavior: `scripts/`

### Adding a new AgPd model variant

When a new AgPd model identity is added, check the following places:

- `src/mkm/models/agpd_registry.py`: one declarative model specification; `src/mkm/models/agpd_basic.py`: prior/parameterization resolution and model assembly;
- `config/models/agpd_basic.yaml`: composition-parameterization entries when the new model differs from its parent;
- `config/priors/agpd_basic.yaml`: canonical prior profiles and named prior-variant overrides;
- `src/mkm/postprocessing/drc.py`: transition-state-control alias when the new model shares an existing mechanism;
- `scripts/run_agpd_individual_grid.py`: runner selector/comparison membership when the model should be included in automated all-material runs;
- `tests/test_agpd_model_registry.py` and any model-specific tests: update exact model lists and test the new behavior;
- search the repository for explicit enumerations of sibling model names, since not every model-specific mapping is derived from the central registry.

## PtRu processed-data contract

The PtRu data are organized separately by medium:

```text
data/processed/PtRu_COOx_acidic/
data/processed/PtRu_COOx_basic/
```

### Todd products

For either medium, let `<dataset>` denote `PtRu_COOx_acidic`
or `PtRu_COOx_basic`.

Standardized observations:

```text
standardized/<dataset>_replicates.parquet
```

Full interpolated analysis observations:

```text
analysis/<dataset>_full.parquet
```

Kinetic observables:

```text
analysis/<dataset>_alpha_replicates.parquet
analysis/<dataset>_alpha_summary.parquet
analysis/<dataset>_delta_CO.parquet
analysis/<dataset>_delta_Hplus.parquet  (acidic only)
analysis/<dataset>_delta_OH.parquet     (basic only)
```

The analysis data use a 0.01 V SHE grid without extrapolation.

Important columns include:

```text
medium
material
C_electrolyte_M
CO_mole_fraction
replicate
analysis_grid_index
E_V_SHE
rate_s_inv
ln_rate
```

Some experimental conditions have different potential support.
The resulting analysis tables are intentionally non-rectangular.

### Em products

Em Pt/C datasets retain distinct sample preparations.
For each `<sample_id>`, the output prefix is:

```text
<dataset>_Em_<sample_id>
```

Each sample has standardized and full analysis Parquets,
alpha replicate and summary Parquets, and a CO-order Parquet.

Em does not produce H+ or OH- orders because its processed
datasets contain only one electrolyte concentration.

Todd and Em data are not combined during preprocessing.

### Reaction-order uncertainty

Where replicate data exist, reaction-order SDs are propagated
from within-condition log-rate variability under an independence
assumption across experimental conditions.

Missing replicate uncertainty remains NaN.

These propagated SDs describe replicate variability rather
than confidence intervals on fitted mean reaction orders.

### Figure hierarchy

Todd figures:

```text
figures/preprocessing/<dataset>/<material>/
```

Em figures:

```text
figures/preprocessing/<dataset>/Pt100_Em_<sample_id>/
```

Each source generates rate and observable figures in PNG and SVG.

Figure filenames and directory structure are preserved by the
PtRu organizational refactor.

### Status

PtRu selected datasets are implemented; only Todd acidic Pt100 currently has exclusions.
Selected Todd observations support individual fits of Pt100, Pt66Ru33, and
Pt50Ru50, plus all-material inference within either acidic or basic media.
Em fitting and pooled PtRu LOCO are not implemented. Pooled PtRu LOMO supports
`shared` or `linear_xRu` physical parameters with shared errors; BF-only LOMO is
unsupported. Model interpretation requires scientific evaluation beyond execution.

## PtRu data selection and selected-observable products

### Ownership

`config/preprocessing/ptru_todd.yaml` defines the selection criteria.

`src/mkm/preprocessing/ptru_selection.py` implements
`select_ptru_analysis()`, returning:

- Selected experimental observations.
- Condition-specific truncation metadata.

`scripts/process_ptru_todd.py` generates the full, selected,
and truncation Parquets.

`scripts/plot_ptru_todd.py` uses the full data and truncation
metadata to generate rate figures with excluded regions visible.

`scripts/process_ptru_observables.py` generates both full and
selected kinetic observables.

### Current selection rules

Only Todd acidic Pt100 has low-potential truncation.

| CO | Criterion |
|---|---|
| 0.1% | 0.70 V SHE at 0.10/0.50 M; 0.75 V SHE at 0.25/1.00 M |
| 1% | First potential reaching 10% of maximum TOF |
| 10% | First potential reaching 5% of maximum TOF |
| 100% | First potential reaching 5% of maximum TOF |

For relative-rate thresholds, the maximum is evaluated
separately for each electrolyte concentration and CO pressure
using the full condition-mean rate curve.

Points at or above the cutoff are retained.

All other Todd materials and media remain untruncated.
Em data remain separate and untruncated.

### Todd processed-data products

For `<dataset>` equal to `PtRu_COOx_acidic` or
`PtRu_COOx_basic`, the analysis directory contains:

```text
<dataset>_full.parquet
<dataset>_selected.parquet
<dataset>_truncation.parquet

<dataset>_alpha_replicates.parquet
<dataset>_alpha_summary.parquet
<dataset>_alpha_replicates_selected.parquet
<dataset>_alpha_summary_selected.parquet

<dataset>_delta_CO.parquet
<dataset>_delta_CO_selected.parquet
```

Acidic Todd additionally produces:

```text
<dataset>_delta_Hplus.parquet
<dataset>_delta_Hplus_selected.parquet
```

Basic Todd instead produces:

```text
<dataset>_delta_OH.parquet
<dataset>_delta_OH_selected.parquet
```

The full analysis dataset is preserved for provenance.

### Observable calculation conventions

Alpha is calculated by differentiating the full experimental
curves and subsequently applying the selected-potential mask.

CO reaction orders are independently recalculated from full
and selected rate data.

H+ and OH- orders are independently recalculated from full
and selected rates, requiring common potential support across
all four electrolyte concentrations.

The original full-data observable Parquets remain available.

### Figure conventions

- Linear TOF: excluded data gray, selected data colored.
- Logarithmic TOF: same.
- Alpha: selected data only.
- CO reaction orders: selected data only.
- H+ and OH- reaction orders: selected data only.

The existing figure filenames and directory hierarchy remain
unchanged.

### Verification status

The selection workflow has synthetic regression tests in
`tests/preprocessing/test_ptru_selection.py` and
`tests/preprocessing/test_ptru_truncation_preview.py`. Comprehensive
parity against archived experimental preprocessing artifacts is not established.

## AgPd data contract

This file documents the AgPd basic-media data contract from raw workbooks through the arrays consumed by PyMC.

The authoritative preprocessing configuration is `config/preprocessing/agpd_basic.yaml`.

### Experimental design

Materials:

```text
Pd100
Ag10Pd90
Ag25Pd75
Ag50Pd50
Ag75Pd25
Ag90Pd10
```

KOH concentrations:

```text
0.25 M
0.50 M
1.00 M
```

CO mole fractions:

```text
0.001
0.01
0.10
1.00
```

Replicates:

```text
A
B
C
```

Potential reference is SHE. Raw AgPd curves are expected to contain 1000 acquisition points and store `ln(j)`, with underlying current density in `uA/cm2` on a Pd-ECSA basis.

### Paired CO-series contract

CO reaction-order comparisons are paired within

```text
(material, C_KOH_M, replicate)
```

KOH concentrations are independent experimental setups. Replicate A is paired to A across CO mole fractions within one material/KOH setup, and likewise for B and C.

### Standardization and interpolation

The preprocessing workflow:

1. loads and validates the raw workbooks;
2. preserves acquisition order separately from analysis-grid order;
3. rejects invalid/non-monotonic potential behavior according to the preprocessing validators;
4. interpolates onto the canonical analysis grid;
5. does not extrapolate outside the measured potential support;
6. calculates rates and derived observables;
7. applies low-potential truncation.

The canonical analysis grid has:

```text
spacing: 0.010 V
origin:  0.0 V
```

Interpolation is performed in log-current space.

### Rate conversion

The rate normalization is Pd-ECSA based:

$$
r=\frac{j}{420}\;\mathrm{s^{-1}},
$$

for `j` in `uA/cm2_Pd`.

The processed analysis tables contain both positive `rate_s_inv` and

$$
\texttt{ln\_rate}=\ln(\texttt{rate\_s\_inv}).
$$

`build_model_data()` validates this equality numerically.

### Truncation

Low-potential selection uses a rate threshold of

$$
r\ge10^{-3}\;\mathrm{s^{-1}}.
$$

All replicates are required to satisfy the threshold at the retained start. There is currently no high-potential truncation.

The exact truncation implementation and metadata live in the preprocessing code and `AgPd_COOx_basic_truncation.parquet`.

### Generated processed files

`process_agpd_basic.py` writes:

```text
data/processed/AgPd_COOx_basic/standardized/
  AgPd_COOx_basic_replicates.parquet

data/processed/AgPd_COOx_basic/analysis/
  AgPd_COOx_basic_full.parquet
  AgPd_COOx_basic_selected.parquet
  AgPd_COOx_basic_summary.parquet
  AgPd_COOx_basic_truncation.parquet
  AgPd_COOx_basic_delta_OH.parquet
  AgPd_COOx_basic_delta_CO_replicates.parquet
  AgPd_COOx_basic_delta_CO.parquet
```

`AgPd_COOx_basic_selected.parquet` is the canonical input to posterior fitting.

### Minimum inference columns

`build_model_data()` requires the selected replicate table to contain:

```text
material
C_KOH_M
CO_mole_fraction
replicate
analysis_grid_index
E_V_SHE
rate_s_inv
ln_rate
```

Requirements:

- electrolyte concentration > 0;
- CO mole fraction > 0;
- rate > 0;
- all numerical fields finite;
- `ln_rate == log(rate_s_inv)` within numerical tolerance;
- no duplicate `(material, KOH, CO, replicate, analysis_grid_index)` observations;
- one potential per condition/grid-index pair.

### Canonical model tables

The selected replicate table is converted into three non-rectangular canonical tables.

### `conditions`

One row per observed

```text
(material, electrolyte_concentration_M, CO_mole_fraction)
```

with a contiguous `condition_id`.

### `model_points`

One row per observed condition/potential point:

```text
model_point_id
condition_id
analysis_grid_index
E_V_SHE
material
electrolyte_concentration_M
CO_mole_fraction
```

The model is evaluated once per `model_point_id`.

### `observations`

One row per replicate observation:

```text
observation_id
model_point_id
condition_id
material
electrolyte_concentration_M
CO_mole_fraction
replicate
analysis_grid_index
E_V_SHE
rate_s_inv
ln_rate
```

Multiple replicate observations may map to one model point. The likelihood mean is therefore evaluated at the model-point level and indexed back to observation rows.

This non-rectangular representation is intentional: only combinations actually present in the processed dataset are represented.

### Arrays passed to PyMC

`ModelInputArrays` contains:

```text
materials
condition_material_index
condition_ln_electrolyte_concentration
condition_ln_CO_mole_fraction
model_point_condition_index
model_point_E_V_SHE
observation_model_point_index
observation_rate
```

`ModelPointInputs` then expands condition-level material/concentration information onto model points and supplies the mechanism with:

```text
materials
material_index
E_V_SHE
ln_electrolyte_concentration
ln_CO_mole_fraction
```

### Derived experimental observables

The preprocessing path produces experimental products used in posterior comparison:

- transfer coefficient `alpha`;
- OH reaction order `delta_OH`;
- adjacent CO reaction order `delta_CO`;
- replicate-resolved adjacent CO-order products.

Posterior observables are calculated from fixed maps applied to posterior model log-rate draws, then compared against these experimental products.

### Validation splits

LOCO holds out exactly one material/KOH/CO condition including all its replicate observations and potential points.

LOMO holds out every observation from one material.

Both splits are made from the canonical selected replicate dataset before rebuilding the three model-data tables for training and held-out prediction.

