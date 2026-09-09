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
src/mkm/models/agpd_basic.py    src/mkm/mechanisms/agpd_basic.py
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

### `src/mkm/mechanisms/agpd_basic.py`

Owns the chemical mathematics:

- activities and surface-composition state;
- equilibrium constants and TST rate constants;
- electrochemical free-energy/barrier conventions;
- Pd-OH and Ag-OH QEA coverages;
- reversible finite-rate CO adsorption/desorption;
- analytic CO steady-state solver;
- BF, ER, LH pathway rates;
- stable pathway log-sum-exp;
- coverage-cap variants;
- Ag10 BF-activity masking;
- mechanism parameter/result dataclasses.

This file should contain chemistry, not PyMC prior choices or filesystem behavior.

### `src/mkm/mechanisms/pd_basic.py`

Owns the reduced pure-Pd ER+LH evaluator used for prediction-only pure-Pd states in all-material validation.

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

Sampling/reconstruction utilities used by the posterior lifecycle.

## Workflow orchestration

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

Owns checkpoint/sample/finalize behavior, including `--resume`, `--overwrite`, metadata, sampler health, and saved posterior lifecycle state.

### `src/mkm/workflows/agpd_validation.py`

Owns LOCO/LOMO split logic and held-out prediction:

- split selected data;
- enforce error-structure support;
- construct prediction-only mechanism graphs;
- compute held-out log predictive density and PIT;
- summarize held-out model/predictive distributions.

## Postprocessing

### `src/mkm/postprocessing/diagnostics.py`

Parameter summaries, sample flattening, physical checks.

### `sampling.py`

Sampler parameter names and sampler-health plotting helpers.

### `predictions.py`

Observation-level posterior mechanism and posterior-predictive diagnostics.

### `residuals.py`

Residual-curve structure and shared-vs-replicate summaries.

### `observables.py` / `observable_comparison.py`

Posterior linear observables and comparison to experimental alpha/OH-order/CO-order products.

### `loo.py`

Single-model PSIS-LOO and pointwise Pareto-k diagnostics.

### `calibration.py`

Normal-model LOO-PIT calculations.

### `model_comparison.py`

Multi-model ELPD comparison and pointwise differences.

### `drc.py`

Individual and composition-resolved transition-state DRC and finite-step convergence checks.

### `composition_parameters.py`

Effective physical-parameter trends versus composition.

### `materials.py`

Material-level aggregation of residual/LOO/observation diagnostics.

### `validation.py`

Validation posterior-shift summaries and held-out validation plots.

### `plotting.py`

Shared plotting implementation used by posterior, DRC, and validation scripts.

## Canonical scripts

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

## Result hierarchy

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
sampling_health.*
tables/
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

- raw workbook parsing: `src/mkm/preprocessing/agpd_basic.py`
- preprocessing facts: `config/preprocessing/agpd_basic.yaml`
- interpolation/truncation: `src/mkm/preprocessing/`
- experimental alpha/orders: `src/mkm/preprocessing/observables.py`
- model data indexing: `src/mkm/model_data.py`, `src/mkm/model_inputs.py`
- a mechanism equation: `src/mkm/mechanisms/agpd_basic.py`
- pathway parameter dataclasses: `src/mkm/mechanisms/agpd_basic.py`
- model registry / special variants: `src/mkm/models/agpd_basic.py`
- base priors / slope priors / likelihood priors: `config/models/agpd_basic.yaml`
- PyMC prior factory: `src/mkm/inference/priors.py`
- likelihood equation: `src/mkm/inference/likelihoods.py`
- fit-scope rules: `src/mkm/workflows/agpd_fit.py`
- posterior provenance/load rules: `src/mkm/workflows/agpd_posterior.py`
- sampling lifecycle: `src/mkm/workflows/posterior_lifecycle.py`
- LOCO/LOMO splitting/prediction: `src/mkm/workflows/agpd_validation.py`
- posterior diagnostics/plots: `src/mkm/postprocessing/`
- canonical CLI behavior: `scripts/`

## Data contract

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
\texttt{ln_rate}=\ln(\texttt{rate_s_inv}).
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