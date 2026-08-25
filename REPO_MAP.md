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
             +----------+----------+
             |                     |
             v                     v
fit_agpd_posterior.py   fit_agpd_composition_posterior.py
             |                     |
             v                     v
       posterior.nc            posterior.nc
             |                     |
             v                     v
postprocess_agpd_posterior.py   postprocess_agpd_composition_posterior.py
             |                     |
             +----------+----------+
                        |
                        v
              src/mkm/postprocessing/*
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
  - likelihood priors
  - material/model prior profiles
  - composition parameterizations and slope priors

These files intentionally describe different concerns and should not be merged.

### `src/mkm/preprocessing/`

Workbook ingestion, validation, interpolation, rate conversion, truncation, experimental alpha, OH order, and paired CO order.

### `src/mkm/model_data.py`

Builds canonical:

- `conditions`
- `model_points`
- `observations`

Only actually observed combinations are represented.

### `src/mkm/model_inputs.py`

Converts canonical tables into indexed NumPy arrays for PyMC/PyTensor, including:

- material/condition/model-point indices
- observation-to-model-point indices
- setup indices retained for the optional setup-intercept likelihood
- model-point mechanism inputs

### `src/mkm/mechanisms/`

Mathematical/chemical kernels:

- thermodynamic and kinetic primitives
- site balances
- QEA coverages
- CO SSA
- pathway rates
- detailed-balance relationship for reversible CO adsorption/desorption
- mechanism parameter/result dataclasses

### `src/mkm/models/`

Fit-ready model registry and composition wrapper:

- parameter dataclass
- mechanism evaluator
- prior-profile compatibility
- individual-material binding
- shared-composition binding
- `linear_xAg` effective-parameter construction

The current linear composition convention is

\[
p(x_{\mathrm{Ag}})=p_{0.5}+s_p(x_{\mathrm{Ag}}-0.5).
\]

### `src/mkm/inference/`

Generic Bayesian machinery:

- prior creation
- IID and setup-intercept log-rate likelihoods
- PyMC model assembly
- posterior sampling
- deterministic reconstruction
- log-likelihood calculation
- prior predictive calculations

The current working likelihood is material-specific IID Normal noise in log-rate space. `setup_intercept` is retained but is not the current default scientific workflow.

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

Individual-material workflow:

- `process_agpd_basic.py`
- `plot_agpd_basic.py`
- `check_agpd_prior_predictive.py`
- `fit_agpd_posterior.py`
- `postprocess_agpd_posterior.py`
- `compare_agpd_models.py`
- `postprocess_agpd_drc.py`

Composition workflow:

- `fit_agpd_composition_posterior.py`
- `postprocess_agpd_composition_posterior.py`
- `compare_agpd_composition_models.py`

See `scripts/README.md` for ownership and limitations.

## Single-model versus multi-model ownership

### Individual-material posterior postprocessing

Owned by:

```text
scripts/postprocess_agpd_posterior.py
```

### Composition posterior postprocessing

Owned by:

```text
scripts/postprocess_agpd_composition_posterior.py
```

Includes material-resolved predictions, residuals, observables, physical variables, and pointwise LOO contributions from a shared/composition-dependent fit.

### Multi-model comparison

Owned by:

```text
scripts/compare_agpd_models.py
scripts/compare_agpd_composition_models.py
```

Comparison outputs are predictive summaries, not mechanism probabilities.

### Transition-state DRC

Owned by:

```text
scripts/postprocess_agpd_drc.py
```

The current implementation is individual-material oriented. Composition-specific effective transition-state energies still need to be added before DRC is applied to `linear_xAg` fits.

## Where do I change...?

- **Raw workbook parsing:** `src/mkm/preprocessing/agpd_basic.py`
- **Validation contracts:** `src/mkm/preprocessing/validation.py`
- **Interpolation/grid rules:** `src/mkm/preprocessing/agpd_basic.py`, `potential.py`
- **Experimental reaction orders:** `src/mkm/preprocessing/observables.py`
- **A mechanism equation:** `src/mkm/mechanisms/agpd_basic.py`
- **Registered models/composition parameterization:** `src/mkm/models/agpd_basic.py`
- **Priors/slopes:** `config/models/agpd_basic.yaml`, `src/mkm/inference/priors.py`
- **Likelihood:** `src/mkm/inference/likelihoods.py`
- **Full PyMC assembly:** `src/mkm/inference/model.py`
- **Posterior sampler:** `src/mkm/inference/posterior.py`
- **Alpha/order maps:** `src/mkm/observable_maps.py`
- **Posterior diagnostics:** `src/mkm/postprocessing/`
- **Plotting:** `src/mkm/postprocessing/plotting.py`
- **Transition-state DRC:** `src/mkm/postprocessing/drc.py`
- **Canonical CLI behavior:** `scripts/`

## AgPd rate/composition convention

Processed rates are normalized per Pd-ECSA-equivalent site. Explicit `Ag_fraction` and `Pd_fraction` factors in the BF/LH pathways represent random-mixing ensemble/neighbor probabilities relative to that Pd-centered normalization; they are not a second ECSA normalization.

The assumptions

```text
surface_composition_equals_bulk: true
random_mixing: true
```

are fixed model assumptions.

## Result hierarchy

Individual-material posterior:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
```

Composition posterior:

```text
results/AgPd_COOx_basic/posterior/composition/<composition_model>/<likelihood>/<model>/
```

Per-model postprocessing:

```text
postprocessing/
├── tables/
├── derived/
└── figures/
```

## Predictive-validation roadmap

Observation-wise PSIS-LOO remains useful for locating where a fit succeeds or fails along measured curves.

Two stronger refit-based validation levels are planned:

- **LOCO:** hold out one `(material, KOH, CO)` experimental condition and all three A/B/C replicate curves;
- **LOMO:** hold out one material/composition entirely.

LOCO tests interpolation/generalization across experimental conditions within a known material. LOMO tests whether composition-dependent energetics generalize to an unseen alloy composition.

## Current refactor priorities

1. keep sampler diagnostics complete for multi-material nuisance parameters
2. update model-comparison CLI to compare `shared` and `linear_xAg` directly
3. add composition-aware prior predictive checks
4. add energy-vs-composition posterior products
5. add composition-specific transition-state DRC
6. add LOCO/LOMO validation workflows
7. split composition postprocessing orchestration after correctness changes
8. define generated-results versioning policy
