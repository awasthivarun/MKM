# AgPd workflow scripts

All commands below are intended to be run from the repository root with the `mkm` environment active.

The CLI scripts are deliberately thin. Reusable scientific/inference logic belongs under `src/mkm/`.

## Preprocessing

Build all standardized and analysis parquet products:

```powershell
python scripts/process_agpd_basic.py
```

Generate experimental preprocessing figures:

```powershell
python scripts/plot_agpd_basic.py
```

## Prior predictive checks

Individual fit identity:

```powershell
python scripts/check_agpd_prior_predictive.py CO_BF_ER --material Ag50Pd50
```

All-material linear-composition identity:

```powershell
python scripts/check_agpd_prior_predictive.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material
```

Use `--save-draws` only when the raw prior-predictive InferenceData is needed.

## Posterior fitting

### Individual material

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50
```

Common sampler override:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --target-accept 0.95 --overwrite
```

Individual fits always use the material error structure.

### All materials, shared physical parameters

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization shared --error-structure material
```

### All materials, linear composition dependence

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material
```

For a full-data run intended as the source/reference for **LOMO**, fit the same model with shared errors:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure shared
```

LOMO cannot extrapolate independent material-specific error parameters to a material omitted from training.

### Resume versus overwrite

Resume finalization from an existing `posterior.nc` checkpoint:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --resume
```

Delete the existing run directory and sample again:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --overwrite
```

## Posterior postprocessing

Individual:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --plot-level full
```

All-material:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --plot-level full
```

Plot levels:

```text
none
core
full
```

Optional switches:

```text
--skip-loo
--skip-observables
```

## Transition-state DRC

Individual:

```powershell
python scripts/postprocess_agpd_drc.py CO_BF_ER --material Ag50Pd50 --check-half-step
```

All-material composition fit:

```powershell
python scripts/postprocess_agpd_drc.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --check-half-step
```

Useful options:

```text
--step-eV <value>
--check-half-step
--save-draws
--skip-plots
```

## Composition parameter trends

```powershell
python scripts/plot_agpd_composition_parameters.py CO_BF_ER_LH --parameterization linear_xAg --error-structure material
```

This writes the effective parameter trend parquet and an overview figure under the fit's `composition/` directory.

## Individual model-grid runner

Run the complete configured individual grid. The script fits each run, applies full posterior postprocessing, and then DRC:

```powershell
python scripts/run_agpd_individual_grid.py
```

Overwrite all existing grid entries:

```powershell
python scripts/run_agpd_individual_grid.py --overwrite
```

Stop at the first failed stage:

```powershell
python scripts/run_agpd_individual_grid.py --stop-on-error
```

The alloy grid contains the seven ordinary pathway subsets. Pd100 runs only `CO_LH`, `CO_ER`, and `CO_ER_LH`.

## Model comparison

### Individual material

Each `--run` is `LABEL,MODEL`:

```powershell
python scripts/compare_agpd_models.py --comparison-name Ag50Pd50_all_models --material Ag50Pd50 --run "CO_BF,CO_BF" --run "CO_BF_ER,CO_BF_ER" --run "CO_BF_ER_LH,CO_BF_ER_LH"
```

Add as many completed candidate runs as needed.

### All-material comparison

Use `--all-materials`. Each `--run` is:

```text
LABEL,MODEL,PARAMETERIZATION,ERROR_STRUCTURE
```

Example:

```powershell
python scripts/compare_agpd_models.py --comparison-name shared_vs_linear --all-materials --run "shared,CO_BF_ER_LH,shared,material" --run "linear,CO_BF_ER_LH,linear_xAg,material"
```

All compared runs must contain the same ordered observations.

Inspect Pareto-k diagnostics before interpreting ELPD differences. Comparison weights are not probabilities that a mechanism is true.

## LOCO validation

LOCO removes one material-specific `(KOH, CO)` condition including all replicate curves and refits the remaining data.

Example with material-specific errors:

```powershell
python scripts/validate_agpd_posterior.py CO_BF_ER_LH loco --holdout-material Ag50Pd50 --koh-M 0.5 --co-mole-fraction 0.1 --parameterization linear_xAg --error-structure material
```

The requested full-data source run with the same model/parameterization/error structure must already exist.

Use `--overwrite` to rerun an existing validation fit and `--resume` to resume finalization from its posterior checkpoint.

## LOMO validation

LOMO removes one material entirely and refits the remaining compositions.

Because the held-out material has no independently fitted error parameters, LOMO requires `--error-structure shared`:

```powershell
python scripts/validate_agpd_posterior.py CO_BF_ER_LH lomo --holdout-material Ag50Pd50 --parameterization linear_xAg --error-structure shared
```

For Pd100 LOMO, the held-out prediction graph automatically uses the reduced ER+LH pure-Pd evaluator. BF-specific parameters remain part of the training all-material posterior but do not enter the pure-Pd held-out rate.

The validation output records whether the held-out Ag fraction is outside the training composition range.

## Special all-material models

Ag10 with BF disabled:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH_Ag10_no_BF --all-materials --parameterization linear_xAg --error-structure material
```

Coverage-cap variants are registered but are currently intended for exploratory sensitivity analysis rather than the main model path. See `MODEL_REGISTRY.md` and the comments in `config/models/agpd_basic.yaml`.

## PowerShell chaining

Windows PowerShell 5 does not support `&&`. To run postprocessing followed by DRC only if postprocessing succeeds:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --plot-level full; if ($?) { python scripts/postprocess_agpd_drc.py CO_BF_ER --material Ag50Pd50 --check-half-step }
```
