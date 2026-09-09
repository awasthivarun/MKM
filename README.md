# MKM

Bayesian microkinetic modeling of electrochemical CO oxidation.

The actively developed system is CO oxidation on AgPd in basic media. The current rebuild supports the full path from raw electrochemical data to individual-material fits, all-material composition fits, posterior diagnostics, pathway-resolved observables, transition-state degree of rate control, model comparison, and refit-based LOCO/LOMO validation.

## Documentation map

Use these files as the maintained project documentation:

- [`CURRENT_STATE.md`](CURRENT_STATE.md): current implementation, scientific conventions, model registry, priors, parameterizations, and known limitations.
- [`REPO_MAP.md`](REPO_MAP.md): code ownership, end-to-end architecture, data contract, and result hierarchy.
- [`scripts/README.md`](scripts/README.md): command-line cookbook.

When documentation conflicts with code or configuration, the current code/configuration controls. The main scientific configuration files are:

```text
config/preprocessing/agpd_basic.yaml
config/models/agpd_basic.yaml
```

## Current AgPd system

Materials:

```text
Pd100
Ag10Pd90
Ag25Pd75
Ag50Pd50
Ag75Pd25
Ag90Pd10
```

The finite-rate CO models combine reversible CO adsorption/desorption on Pd with Pd-OH and Ag-OH quasi-equilibria and selectable BF, ER, and LH oxidation pathways. The model is evaluated in log space wherever practical; the CO steady state is solved analytically rather than with an iterative solver.

The active likelihood is Normal in **linear rate space**:

$$
r_i^{\mathrm{obs}} \sim \mathcal N\!\left(r_i^{\mathrm{model}},\sigma_i^2\right),
\qquad
\sigma_i=\sigma_{\mathrm{rate,abs}}+\sigma_{\mathrm{rate,rel}}r_i^{\mathrm{model}}.
$$

All-material fits support `shared` and `material` error structures. Individual-material fits use one material-indexed error pair.

All-material composition fits support parameterizations configured in `config/models/agpd_basic.yaml`, including `shared` and the actively used `linear_xAg` model.

## Environment

The validated environment is defined by `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate mkm
python -m pip install -e . --no-deps
pytest tests -q
```

`--no-deps` is intentional when the Conda environment has already installed the validated scientific stack.

## Common workflow

Preprocess and plot the AgPd basic dataset:

```powershell
python scripts/process_agpd_basic.py
python scripts/plot_agpd_basic.py
```

Fit one individual model:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50
```

Run full posterior diagnostics and DRC:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --plot-level full
python scripts/postprocess_agpd_drc.py CO_BF_ER --material Ag50Pd50 --check-half-step
```

Fit the all-material linear-composition model:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material
```

Postprocess it, calculate composition-resolved DRC, and plot effective parameter trends:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --plot-level full
python scripts/postprocess_agpd_drc.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --check-half-step
python scripts/plot_agpd_composition_parameters.py CO_BF_ER_LH --parameterization linear_xAg --error-structure material
```

See [`scripts/README.md`](scripts/README.md) for comparison, prior-predictive, grid, LOCO, LOMO, resume, and overwrite commands.

## Repository layout

```text
config/                  preprocessing and model configuration
data/                    raw and processed experimental data
figures/                 generated preprocessing figures
results/                 posterior, validation, and postprocessing products
scripts/                 thin command-line workflow entry points
src/mkm/                 reusable scientific/inference/postprocessing code
tests/                   unit, integration, workflow, sampling, and plotting tests
CURRENT_STATE.md         current implementation, conventions, and model registry
REPO_MAP.md              architecture, data contract, and code ownership
```

## Interpretation boundary

A converged sampler, narrow posterior, good posterior predictive fit, favorable PSIS-LOO score, or nonzero pathway fraction is not by itself proof of a microscopic mechanism. Physical consistency, parameter identifiability, residual structure, coverages, pathway rates, DRCs, and held-out prediction should be evaluated separately.
