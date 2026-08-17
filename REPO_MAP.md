# Repository Map

## Purpose

This document describes the current organization of the MKM repository and the role of each major directory.

This is a map of the repository before the major architecture refactor.

---

## Root

### `_CO_Oxidation/`

Reusable backend for CO oxidation modeling.

Key modules:

* `config.py` — physical constants and acid/base-specific configuration
* `core.py` — shared mutable analysis state
* `context.py` — combines processing, plotting, posterior, and DRC functionality
* `processing.py` — experimental-data loading/flattening and posterior kinetic observables
* `plotting.py` — model-fit and coverage plots
* `posteriors.py` — posterior summaries, prior/posterior comparisons, PPC diagnostics
* `drc.py` — degree-of-rate-control calculations
* `wrapper_base.py` — convenience interface for alkaline models
* `wrapper_acid.py` — convenience interface for acidic models

The current wrappers expose global context objects for notebook use.

### `_plotting_defaults/`

Shared plotting configuration and Matplotlib defaults.

### `AgPd_COOx/`

AgPd CO oxidation project.

Current materials:

* `Ag10Pd90`
* `Ag25Pd75`
* `Ag50Pd50`
* `Ag75Pd25`
* `Ag90Pd10`
* `Pd100`

Also contains:

* `_archive/` — historical calculations and source material
* `_expt_data/` — experimental CSV data
* project-level analysis notebooks

Typical material directory currently contains:

* preprocessing notebook
* broad model-screening notebook
* reduced/finalist model notebook
* executed HTML report
* processed experimental pickle
* supporting spreadsheets or other material-specific outputs

### `PtRu_COox/`

PtRu CO oxidation project.

Contains material/environment-specific analyses including combinations of:

* Pt100
* Pt50Ru50
* Pt66Ru33
* acidic electrolyte
* alkaline electrolyte

Also contains project-level plots and summary spreadsheets.

### `HER_HOR/`

Separate HER/HOR experimental-data work.

Not part of the immediate CO oxidation refactor.

### `test/`

Currently contains older/smaller analysis examples.

This is not yet the repository's automated software-test suite.

The future automated test suite should use a clearly defined `tests/` directory.

### `.vscode/`

Local VS Code configuration.

Most editor-specific content is excluded by `.gitignore`.

---

## Root documentation

### `workflow_COOx.md`

Scientific/computational workflow from experimental preprocessing through Bayesian model fitting and diagnostics.

### `DATA_CONTRACT.md`

Formal definition of CO oxidation processed-data structures, dimensions, units, indexing, and planned mega-model metadata.

### `MODEL_REGISTRY.md`

Planned catalog of kinetic models, assumptions, parameters, likelihoods, and scientific status.

### `CURRENT_STATE.md`

Short handoff document describing the current scientific and software-development state.

### `README.md`

Planned repository entry point for setup, navigation, and basic usage.

### `REPO_MAP.md`

This file.

---

## Environment snapshots

### `environment_current.yml`

Portable description of the current working Conda environment.

### `environment_current_explicit.txt`

More exact package-level snapshot of the pre-refactor environment.

The current environment should be preserved for reproducibility rather than upgraded in place.

---

## Git/refactor records

### `repo_tree_before_refactor.txt`

Snapshot of the directory tree immediately before the major project refactor.

### Active development branches

* `main` — stable pre-refactor project state
* `refactor/project-architecture` — active architecture refactor

---

## Current architectural boundary

At present:

* reusable numerical/plotting functionality lives mainly in `_CO_Oxidation/`;
* kinetic model definitions live primarily inside notebooks;
* processed data generally live inside material-specific directories;
* inference orchestration is repeated across notebooks;
* individual-material analysis is the dominant organizational unit.

The refactor will gradually separate:

1. reusable software,
2. experimental/processed data,
3. model definitions,
4. notebooks,
5. generated results and reports,
6. automated tests.
