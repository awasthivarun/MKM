# MKM

Bayesian microkinetic modeling of electrochemical CO oxidation.

The rebuilt workflow is currently most developed for AgPd catalysts in basic media. All six AgPd compositions are represented in the processed dataset, individual-material and shared-composition posterior workflows are implemented, and the current composition model supports selected energetic parameters that vary linearly with Ag fraction.

## Current status

- Active development branch: `rebuild/from-scratch`
- Current scientific system: AgPd CO oxidation in basic media
- Registered AgPd mechanisms:
  - `BF`
  - `BF_LH`
  - `CO_BF_ER_LH`
- Current working likelihood:
  - material-specific IID Normal residuals in log-rate space
- Retained alternative likelihood:
  - zero-sum setup-intercept likelihood, currently not the default scientific workflow
- Composition parameterizations:
  - `shared`
  - `linear_xAg`
- Permanent posterior postprocessing, PSIS-LOO, LOO-PIT, model comparison, and transition-state DRC are implemented.
- Composition-specific DRC and grouped cross-validation are not yet implemented.

See:

- [`CURRENT_STATE.md`](CURRENT_STATE.md) for the scientific/development checkpoint
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
- `src/mkm/models/` binds mechanism evaluators and parameter dataclasses to configured prior profiles and composition parameterizations.
- `src/mkm/inference/` assembles and samples chemistry-independent PyMC models.
- `src/mkm/postprocessing/` calculates posterior diagnostics, predictive validation, model comparison, and DRC.
- `scripts/` should remain thin workflow entry points; reusable calculations belong in `src/mkm/`.

## Canonical workflow

### Preprocess AgPd basic data

```powershell
python scripts/process_agpd_basic.py
python scripts/plot_agpd_basic.py
```

### Individual-material prior predictive checks

```powershell
python scripts/check_agpd_prior_predictive.py
```

A composition-aware prior-predictive workflow is still to be added.

### Fit an individual-material posterior

Use the likelihood explicitly rather than relying on defaults:

```powershell
python scripts/fit_agpd_posterior.py BF_LH --likelihood iid
```

### Fit a shared-composition posterior

```powershell
python scripts/fit_agpd_composition_posterior.py CO_BF_ER_LH --composition-model shared --likelihood iid
```

### Fit linear Ag-composition energetics

```powershell
python scripts/fit_agpd_composition_posterior.py CO_BF_ER_LH --composition-model linear_xAg --likelihood iid
```

The current parameterization is

\[
p(x_{\mathrm{Ag}})=p_{0.5}+s_p(x_{\mathrm{Ag}}-0.5).
\]

Only parameters listed under `composition_parameterizations.linear_xAg` in `config/models/agpd_basic.yaml` receive composition slopes.

### Postprocess an individual-material posterior

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood iid
```

### Postprocess a composition posterior

```powershell
python scripts/postprocess_agpd_composition_posterior.py CO_BF_ER_LH --composition-model linear_xAg --likelihood iid
```

Persistent products include sampler diagnostics, parameter summaries, posterior rate/residual products, physical variables, experimental observable comparisons, pointwise PSIS-LOO, Pareto-k, and LOO-PIT diagnostics.

### Compare models

Individual-material comparison:

```powershell
python scripts/compare_agpd_models.py --likelihood iid
```

Composition-model comparison exists, but direct `shared` versus `linear_xAg` comparison still needs a small CLI update. Always specify the likelihood explicitly.

### Compute transition-state DRC

```powershell
python scripts/postprocess_agpd_drc.py BF_LH --likelihood iid --check-half-step
```

Current DRC is transition-state-only. Composition-specific effective transition-state energies are not yet wired into the DRC workflow.

## Output convention

Individual-material posterior outputs:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
```

Composition posterior outputs:

```text
results/AgPd_COOx_basic/posterior/composition/<composition_model>/<likelihood>/<model>/
```

Per-model postprocessing follows:

```text
postprocessing/
├── tables/
├── derived/
└── figures/
```

## Scientific interpretation boundary

Statistical fit, posterior convergence, narrow HDIs, or LOO ranking do not establish a mechanism by themselves.

Current evidence supports that:

- pure BF is insufficient for the full AgPd problem;
- ER-containing chemistry is required to describe Pd100 within the current model family;
- shared energetics are inadequate across the alloy series;
- a linear-in-Ag energetic parameterization can be sampled cleanly after excluding implausible remote prior modes;
- strong residual correlation along potential remains under the IID likelihood;
- CO reaction-order behavior remains an important model-data mismatch;
- Ag10Pd90 remains notably difficult for the current shared/composition model family.

Future predictive validation should distinguish:

- **LOCO:** leave one material-specific `(KOH, CO)` condition out, including all three replicates;
- **LOMO:** leave one material/composition out entirely.

These tests answer stronger scientific questions than observation-wise PSIS-LOO.
