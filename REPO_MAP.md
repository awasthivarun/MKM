# REPO_MAP

Detailed orientation map for the MKM rebuild repository.

## Big-picture workflow graph

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
        +-----------------------------+
        |                             |
        v                             v
src/mkm/models/agpd_basic.py   src/mkm/mechanisms/agpd_basic.py
        |                             |
        +--------------+--------------+
                       |
                       v
              src/mkm/inference/model.py
                       |
                       v
           src/mkm/inference/likelihoods.py
                       |
                       v
             posterior fitting scripts
                       |
                       v
                  posterior.nc
                       |
                       v
              diagnostics / plots / LOO
```

## Core directories and responsibilities

- `config/preprocessing/agpd_basic.yaml`: experimental-data and preprocessing conventions (materials, concentrations, pairing structure, grid/truncation rules).
- `config/models/agpd_basic.yaml`: model-facing configuration (surface composition assumptions, priors, likelihood hyperparameters).
- `data/raw/AgPd_COOx_basic/`: source workbooks.
- `data/processed/AgPd_COOx_basic/`: standardized/analysis parquet outputs used by inference.
- `src/mkm/preprocessing/`: workbook parsing, interpolation, truncation, derived experimental observables.
- `src/mkm/model_data.py`: converts selected replicate tables into canonical `conditions`, `model_points`, `observations` tables.
- `src/mkm/model_inputs.py`: converts canonical tables into indexed arrays and coordinates for PyMC/PyTensor.
- `src/mkm/mechanisms/`: mechanism equations and evaluators.
- `src/mkm/models/`: model registry and mechanism/prior-profile binding.
- `src/mkm/inference/`: priors, likelihood construction, model assembly, posterior and prior-predictive utilities.
- `src/mkm/observable_maps.py`: linear maps from modeled log-rate draws to derived observables (`alpha`, `delta_OH`, adjacent `delta_CO`).
- `scripts/`: workflow/diagnostic entry points.
- `tests/`: validation of preprocessing, mechanisms, model assembly, priors, likelihoods, diagnostics.
- `results/` and `figures/`: generated inference artifacts and plots.

## Where do I change...?

- **Raw workbook parsing**: `src/mkm/preprocessing/agpd_basic.py` (`load_agpd_workbook`, parsing helpers), plus `src/mkm/preprocessing/validation.py` contracts.
- **Interpolation/grid rules**: `src/mkm/preprocessing/agpd_basic.py` (`_build_analysis_grid`, interpolation path in `build_agpd_analysis_grid`).
- **Reaction-order calculations**: `src/mkm/preprocessing/observables.py` and AgPd wrappers in `src/mkm/preprocessing/agpd_basic.py`.
- **A chemical mechanism equation**: `src/mkm/mechanisms/agpd_basic.py`.
- **Add a new mechanism**: mechanism layer first (`src/mkm/mechanisms/`), then register in `src/mkm/models/agpd_basic.py`.
- **Prior distributions**: `config/models/agpd_basic.yaml` prior profiles + interpretation in `src/mkm/inference/priors.py`.
- **Likelihood definition**: `src/mkm/inference/likelihoods.py`.
- **PyMC full-model assembly**: `src/mkm/inference/model.py`.
- **Posterior sampling execution**: `scripts/fit_agpd_posterior.py` using `src/mkm/inference/posterior.py` utilities.
- **Observable maps**: `src/mkm/observable_maps.py`.
- **Future reusable post-processing calculations**: should move toward `src/mkm/` modules; currently much remains in `scripts/` diagnostics.
- **Tests**: `tests/` and `tests/preprocessing/`, `tests/inference/`.

## Mechanisms vs models (important distinction)

- `src/mkm/mechanisms/` = mathematical/chemical kernels and parameter/result dataclasses.
- `src/mkm/models/` = fit-ready registry and compatibility checks that tie mechanism dataclasses/evaluators to configured prior profiles.

This separation allows mechanism development without rewriting generic inference infrastructure.

## Adding a new AgPd mechanism (extensibility path)

For a **new fit configuration of an existing evaluator**:

1. Register the model in `src/mkm/models/agpd_basic.py`.
2. Add prior profile(s) in `config/models/agpd_basic.yaml`.
3. Add tests.
4. Reuse existing generic inference pipeline.

For a **new mathematical mechanism**:

1. Add parameter/result dataclass and evaluator in the mechanism layer.
2. Register it in `src/mkm/models/agpd_basic.py`.
3. Add priors in `config/models/agpd_basic.yaml`.
4. Add mechanism-specific tests.
5. Reuse generic model assembly/likelihood/posterior infrastructure.

This ordinarily should not require changes to preprocessing, model-data construction, model-input indexing, generic likelihood assembly, generic PyMC model assembly, posterior sampling utilities, or existing alpha/OH-order/CO-order map machinery.

## Notes on current script modularity

Inference infrastructure under `src/mkm/inference/` is relatively modular.
Current posterior post-processing in `scripts/` is less modular and includes mixed calculation + plotting code. That is a known future refactor area, not changed during this housekeeping pass.

## Results-path status

Historical outputs currently include both styles:

- `results/AgPd_COOx_basic/posterior/Ag10Pd90/BF`
- `results/AgPd_COOx_basic/posterior/Ag10Pd90/setup_intercept/BF`

Legacy paths are retained for provenance compatibility.

Intended eventual canonical hierarchy:

```text
results/
└── AgPd_COOx_basic/
    └── posterior/
        └── <material>/
            └── <likelihood>/
                └── <model>/
```

Examples:

- `Ag10Pd90/setup_intercept/BF`
- `Ag10Pd90/setup_intercept/BF_LH`
- `Ag10Pd90/setup_intercept/CO_BF_ER_LH`

Path centralization/migration is intentionally deferred to a future housekeeping step to avoid risking historical-result provenance.

## Config split is intentional

- `config/preprocessing/agpd_basic.yaml` = experiment/data-processing facts.
- `config/models/agpd_basic.yaml` = model/statistical conventions and priors.

These are intentionally separate concerns and should not be merged.
