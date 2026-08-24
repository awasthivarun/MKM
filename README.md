# MKM

Bayesian microkinetic modeling of electrochemical CO oxidation.

The rebuilt workflow is currently most developed for AgPd catalysts in basic media, with Ag10Pd90 used as the first complete posterior-analysis case. The package architecture is intended to support the remaining AgPd compositions and later PtRu acid/basic models without returning to notebook-specific implementations.

## Current status

- Active development branch: `rebuild/from-scratch`
- Validated local checkpoint: **192 tests passed**
- Known warnings: **7** environment/runtime warnings, currently treated as non-failing
- Current fitted Ag10Pd90 models:
  - `BF`
  - `BF_LH`
  - `CO_BF_ER_LH`
- Current working likelihood:
  - zero-sum setup-intercept Normal likelihood in log-rate space
- Permanent posterior postprocessing, PSIS-LOO, LOO-PIT, model comparison, and transition-state DRC are implemented.

See:

- [`CURRENT_STATE.md`](CURRENT_STATE.md) for scientific and development status
- [`REPO_MAP.md`](REPO_MAP.md) for the code/workflow map
- [`scripts/README.md`](scripts/README.md) for script ownership and usage

## Environment setup

The validated development environment is specified in [`environment.yml`](environment.yml).

From the repository root:

```powershell
conda env create -f environment.yml
conda activate mkm
python -m pip install -e . --no-deps
pytest tests -q
```

`--no-deps` is intentional when using `environment.yml`: Conda has already installed the validated binary/scientific stack.

A pip-only editable install is also described by `pyproject.toml`:

```powershell
python -m pip install -e ".[dev]"
```

The Conda environment is preferred for the tested PyMC/PyTensor/nutpie/Numba stack.

## Repository layout

```text
config/                  preprocessing and model/likelihood/prior configuration
data/                    raw and processed experimental data
scripts/                 reproducible command-line workflow entry points
src/mkm/                 scientific, inference, and postprocessing implementation
tests/                   unit, integration, sampling, and plotting tests
results/                 generated posterior/postprocessing outputs
figures/                 generated preprocessing figures
CURRENT_STATE.md         detailed project/scientific checkpoint
REPO_MAP.md              architecture and extension guide
environment.yml          validated Conda development environment
pyproject.toml           package metadata, dependencies, and tool configuration
```

## Architecture principle

- `src/mkm/mechanisms/` contains mathematical/chemical mechanism implementations.
- `src/mkm/models/` binds mechanism evaluators and parameter dataclasses to configured prior profiles.
- `src/mkm/inference/` assembles and samples chemistry-independent PyMC models.
- `src/mkm/postprocessing/` calculates posterior diagnostics, predictive validation, model comparison, and DRC.
- `scripts/` should remain thin workflow entry points; reusable calculations belong in `src/mkm/`.

## Canonical workflow

### Preprocess AgPd basic data

```powershell
python scripts/process_agpd_basic.py
python scripts/plot_agpd_basic.py
```

### Prior predictive checks

```powershell
python scripts/check_agpd_prior_predictive.py
```

### Fit a posterior

```powershell
python scripts/fit_agpd_posterior.py BF_LH --likelihood setup_intercept
```

### Generate all single-model posterior products

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood setup_intercept
```

This creates persistent:

- sampler diagnostics
- parameter/posterior summaries
- posterior predictive and residual products
- coverages and pathway fractions
- experimental alpha/OH-order/CO-order comparisons
- pointwise PSIS-LOO and Pareto-k
- LOO-PIT calibration diagnostics

### Compare models

```powershell
python scripts/compare_agpd_models.py --likelihood setup_intercept
```

This contains only genuinely multi-model products:

- `az.compare` / stacking
- aggregate ELPD differences
- pointwise ELPD differences

### Compute transition-state DRC

```powershell
python scripts/postprocess_agpd_drc.py BF_LH --likelihood setup_intercept --check-half-step
```

## Output convention

Canonical posterior outputs follow:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
```

Per-model postprocessing follows:

```text
postprocessing/
├── tables/
├── derived/
└── figures/
```

Historical output paths are retained for provenance; new scripts use the likelihood-aware hierarchy.

## Scientific interpretation boundary

Statistical fit, LOO ranking, or posterior convergence do not by themselves establish a mechanism.

The current Ag10Pd90 results support that:

- BF alone is predictively inadequate;
- added pathway complexity improves some behavior;
- all current ideal mean-field mechanisms retain a severe CO-order mismatch;
- most residual structure is shared across replicates and represents model discrepancy rather than simple experimental noise.

See `CURRENT_STATE.md` for the evidence hierarchy and unresolved hypotheses.
