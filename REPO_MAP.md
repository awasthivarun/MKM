# REPO_MAP

Orientation guide for the MKM rebuild.

## End-to-end workflow

```text
Raw Excel workbooks
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
processed parquet tables
        |
        v
src/mkm/model_data.py
        |
        v
conditions / model_points / observations
        |
        v
src/mkm/model_inputs.py
        |
        +-------------------------------+
        |                               |
        v                               v
src/mkm/models/agpd_basic.py     src/mkm/mechanisms/agpd_basic.py
        |                               |
        +---------------+---------------+
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
        +---------------+----------------+
        |                                |
        v                                v
scripts/postprocess_agpd_posterior.py   scripts/postprocess_agpd_drc.py
        |                                |
        v                                v
src/mkm/postprocessing/*               TS-DRC draws/summaries/checks
        |
        v
single-model tables / derived data / figures
        |
        v
scripts/compare_agpd_models.py
        |
        v
multi-model ELPD differences / stacking
```

## Core directories

### `config/`

- `config/preprocessing/agpd_basic.yaml`
  - experimental design and preprocessing facts
  - materials, concentrations, replicates
  - paired CO-series contract
  - grid, interpolation, normalization, truncation

- `config/models/agpd_basic.yaml`
  - gas/electrolyte/model conventions
  - surface-composition assumptions
  - likelihood priors and setup grouping
  - material/model prior profiles

These files intentionally describe different concerns and should not be merged.

### `src/mkm/preprocessing/`

Workbook ingestion, validation, interpolation, rate conversion, truncation, experimental alpha, OH order, and paired CO order.

### `src/mkm/model_data.py`

Builds canonical:

- `conditions`
- `model_points`
- `observations`

This is the stable boundary between dataset-specific preprocessing and generic modeling.

### `src/mkm/model_inputs.py`

Converts canonical tables into indexed NumPy arrays for PyMC/PyTensor, including:

- material/condition/model-point indices
- observation-to-model-point indices
- setup indices
- zero-sum setup-experiment indices
- model-point mechanism inputs

### `src/mkm/mechanisms/`

Mathematical/chemical kernels:

- thermodynamic and kinetic primitives
- site balances
- QEA coverages
- CO SSA
- pathway rates
- mechanism parameter/result dataclasses

### `src/mkm/models/`

Fit-ready model registry:

- parameter dataclass
- mechanism evaluator
- prior-profile compatibility
- material/model binding

### `src/mkm/inference/`

Generic Bayesian machinery:

- prior creation
- log-rate likelihoods
- PyMC model assembly
- posterior sampling
- deterministic reconstruction
- log-likelihood calculation
- prior predictive calculations

### `src/mkm/observable_maps.py`

Fixed linear maps from model log rate to:

- transfer coefficient `alpha`
- OH reaction order
- adjacent CO reaction order

### `src/mkm/postprocessing/`

Numerical posterior products:

- `diagnostics.py`: parameter/noise/physical summaries and balance checks
- `sampling.py`: sampler diagnostics and ArviZ-compatible sampler-stat normalization
- `predictions.py`: observation-level mechanism/conditional/predictive distributions
- `residuals.py`: curve structure and shared-vs-replicate discrepancy
- `observables.py`: pointwise and linear-observable posterior summaries
- `observable_comparison.py`: posterior-vs-experiment comparisons
- `loo.py`: single-model PSIS-LOO and Pareto-k
- `calibration.py`: analytic Normal LOO-PIT
- `model_comparison.py`: genuinely multi-model ELPD comparisons
- `drc.py`: posterior transition-state DRC
- `plotting.py`: current figure implementation

## Canonical scripts

- `process_agpd_basic.py`
- `plot_agpd_basic.py`
- `check_agpd_prior_predictive.py`
- `fit_agpd_posterior.py`
- `postprocess_agpd_posterior.py`
- `compare_agpd_models.py`
- `postprocess_agpd_drc.py`

See `scripts/README.md` for exact ownership.

## Single-model versus multi-model ownership

### Single-model posterior postprocessing

Owned by:

```text
scripts/postprocess_agpd_posterior.py
```

Includes:

- sampler diagnostics
- posterior parameter plots/contraction
- predictions/residuals
- physical checks
- alpha/reaction-order comparisons
- pointwise PSIS-LOO
- Pareto-k
- LOO-PIT and calibration
- coverages/pathway fractions

### Multi-model comparison

Owned by:

```text
scripts/compare_agpd_models.py
```

Includes only quantities that require multiple posteriors:

- model comparison table
- stacking weights
- aggregate ELPD differences
- pointwise ELPD differences

### Transition-state DRC

Owned by:

```text
scripts/postprocess_agpd_drc.py
```

Persists:

- posterior DRC draws
- condition-resolved summaries
- sum-rule checks
- finite-difference step-convergence checks
- DRC figures

## Where do I change...?

- **Raw workbook parsing:** `src/mkm/preprocessing/agpd_basic.py`
- **Validation contracts:** `src/mkm/preprocessing/validation.py`
- **Interpolation/grid rules:** `src/mkm/preprocessing/agpd_basic.py`, `potential.py`
- **Experimental reaction orders:** `src/mkm/preprocessing/observables.py`
- **A mechanism equation:** `src/mkm/mechanisms/agpd_basic.py`
- **Registered models:** `src/mkm/models/agpd_basic.py`
- **Priors:** `config/models/agpd_basic.yaml`, `src/mkm/inference/priors.py`
- **Likelihood:** `src/mkm/inference/likelihoods.py`
- **Full PyMC assembly:** `src/mkm/inference/model.py`
- **Posterior sampler:** `src/mkm/inference/posterior.py`
- **Alpha/order maps:** `src/mkm/observable_maps.py`
- **Posterior diagnostic calculation:** `src/mkm/postprocessing/`
- **Plotting:** `src/mkm/postprocessing/plotting.py`
- **Transition-state DRC definitions:** `src/mkm/postprocessing/drc.py`
- **Canonical CLI behavior:** `scripts/`

## Adding a new AgPd model

### Existing evaluator, new fit-ready model

1. register it in `src/mkm/models/agpd_basic.py`
2. add material/model priors in `config/models/agpd_basic.yaml`
3. add tests
4. reuse the generic fit/postprocessing infrastructure

### New mathematical mechanism

1. add parameter/result dataclasses and evaluator in `src/mkm/mechanisms/`
2. register the model
3. add prior profiles
4. add mechanism/limit/physicality tests
5. declare its pointwise outputs and DRC controls
6. reuse generic inference and observable maps

Normally this should not require changes to:

- preprocessing
- model-data construction
- generic likelihood assembly
- generic PyMC assembly
- posterior sampling
- alpha/OH-order/CO-order maps

## Current extensibility bottleneck

The inference core is modular, but model-specific postprocessing metadata is currently spread across:

- model registry
- pointwise-variable lists
- physical-balance lists
- DRC control registry

The next extensibility refactor should introduce one model metadata contract for:

- pointwise variables
- pathway fractions
- site balances
- transition-state controls

## Result hierarchy

Canonical:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
├── posterior_free.nc
├── posterior.nc
├── sampler_diagnostics.csv
└── postprocessing/
    ├── tables/
    ├── derived/
    └── figures/
```

Multi-model comparison:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/model_comparison/
```

Historical pre-likelihood paths remain for provenance and should be migrated deliberately, not deleted ad hoc.

## Environment

- `pyproject.toml`: package metadata, direct Python dependencies, tool configuration
- `environment.yml`: validated Conda development/scientific stack

Recommended setup:

```powershell
conda env create -f environment.yml
conda activate mkm
python -m pip install -e . --no-deps
```

## Current refactor priorities

1. centralize paths/config/material context
2. add `--material` to canonical scripts
3. move posterior orchestration into package-level workflow functions
4. split postprocessing plotting by domain
5. centralize model metadata
6. add run metadata/provenance
7. define generated-results policy
8. add CLI integration tests
