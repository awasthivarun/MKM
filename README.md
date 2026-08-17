# MKM

Research code for Bayesian microkinetic modeling of electrochemical reactions, primarily CO oxidation on PtRu and AgPd catalysts.

## Current focus

The active development focus is:

- AgPd CO oxidation in alkaline electrolyte
- transition from individual-material models to multi-material "mega-models"
- refactoring the modeling framework for reproducibility, automation, testing, and maintainability

## Repository areas

- `_CO_Oxidation/` — reusable CO oxidation modeling backend
- `AgPd_COOx/` — AgPd experimental data, preprocessing, and model analyses
- `PtRu_COox/` — PtRu experimental data and model analyses
- `_plotting_defaults/` — shared plotting configuration
- `HER_HOR/` — separate HER/HOR work

See `REPO_MAP.md` for a detailed directory description.

## Documentation

- `workflow_COOx.md` — scientific and computational workflow
- `DATA_CONTRACT.md` — processed-data definitions, shapes, units, and indexing
- `MODEL_REGISTRY.md` — mechanistic model catalog
- `CURRENT_STATE.md` — current development/scientific status
- `REPO_MAP.md` — repository structure

## Environment

The pre-refactor working Conda environment is recorded in:

- `environment_current.yml`
- `environment_current_explicit.txt`

Do not upgrade the original reproducibility environment in place. Dependency upgrades should be tested in a separate environment.

## Development status

Stable pre-refactor code is preserved on `main`.

Major architecture work is being developed on:

`refactor/project-architecture`