# Current Project State

Last updated: 2026-08-17

## Scientific stage

### AgPd CO oxidation
- Individual-material model screening is complete for:
  - Ag10Pd90
  - Ag25Pd75
  - Ag50Pd50
  - Ag75Pd25
  - Ag90Pd10
  - Pd100
- Each material has:
  - preprocessing notebook
  - broad model-screening notebook
  - reduced/finalist notebook using the preferred error model
- HTML exports are currently used as frozen records of executed notebooks.
- Next major scientific objective: construct AgPd mega-models that simultaneously fit every material and experimental condition.

### PtRu CO oxidation
- Existing acid/base analyses are present but are not the immediate refactor target.

## Current modeling workflow

1. Preprocess replicate experimental data.
2. Interpolate/truncate data onto working potential grids.
3. Convert current density to TOF.
4. Calculate experimental kinetic observables:
   - alpha
   - delta_CO
   - delta_OH or delta_H
5. Export processed experimental dictionaries as pickle files.
6. Build PyMC microkinetic models in notebooks.
7. Sample models with nutpie.
8. Calculate log likelihood and LOO.
9. Add posterior kinetic observables.
10. Generate posterior, model-fit, coverage, and DRC plots.

## Current backend architecture

Reusable code is primarily in `_CO_Oxidation/`:

- `config.py` — constants and acid/base configuration
- `core.py` — mutable context state
- `processing.py` — experimental-data processing and posterior observables
- `plotting.py` — model-fit and coverage plotting
- `posteriors.py` — posterior/PPC diagnostics
- `drc.py` — degree-of-rate-control calculations
- `context.py` — combines backend mixins
- `wrapper_base.py` / `wrapper_acid.py` — notebook-facing global contexts

## Known architectural limitations

### 1. Single-material indexing
The current backend identifies experimental conditions using:

`(concentration, P_CO)`

This is insufficient for mega-models. Multi-material support must include material identity, conceptually:

`(material, concentration, P_CO)`

### 2. Repeated flattened-array indexing
Processing, plotting, coverage plotting, and DRC independently reconstruct locations in flattened arrays.

A single authoritative condition-to-slice mapping should replace this.

### 3. Global mutable context
`wrapper_base.py` and `wrapper_acid.py` expose one global mutable context each.

This is convenient for existing notebooks but should not be the primary architecture for future multi-material and validation workflows.

### 4. Model definitions live in notebooks
Individual notebooks contain many complete PyMC model definitions.

This creates duplication across materials and makes global model changes difficult to propagate safely.

### 5. Inference logic lives in notebooks
Functions such as model compilation, sampling, LOO calculation, and common likelihood/error-model definitions are repeated in notebooks and should become reusable backend components.

### 6. Notebook naming
The current `base` / `base_error` naming mixes electrolyte identity with analysis stage.

Future naming should distinguish concepts such as:
- preprocessing
- screening
- finalists
- mega-models

## Environment

Current working environment:

`jdftx_env`

The exact pre-refactor environment has been saved as:

- `environment_current.yml`
- `environment_current_explicit.txt`

The current environment should NOT be upgraded in place.

Future dependency upgrades will be tested in a cloned/new environment before adoption.

## Git state at start of refactor

Stable pre-refactor project snapshot:
- branch: `main`
- pushed to GitHub

Active refactor branch:
- `refactor/project-architecture`

An independent external backup of the entire repository also exists.

## Immediate refactor goals

1. Document the current data contract.
2. Document repository structure.
3. Create a model registry.
4. Make the project installable without `sys.path.append(...)`.
5. Add automated regression tests for existing numerical behavior.
6. Introduce explicit material-aware data structures.
7. Introduce one authoritative condition-slice/index mapping.
8. Refactor inference/error-model logic out of notebooks.
9. Refactor repeated mechanism definitions out of notebooks.
10. Automate fitting, diagnostics, result saving, and report generation.
11. Implement AgPd mega-model support.
12. Evaluate dependency upgrades in an isolated environment.