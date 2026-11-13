# MKM workflow scripts

All commands below are intended to be run from the repository root with the `mkm` environment active.

The CLI scripts are deliberately thin. Reusable scientific/inference logic belongs under `src/mkm/`.

## PtRu preprocessing and observables

PtRu preprocessing covers Pt100, Pt66Ru33, and Pt50Ru50 in acidic
and basic media. Todd and Em measurements are processed separately.

The current PtRu pipeline produces experimental rates, apparent
transfer coefficients, CO reaction orders, and H+/OH- reaction orders.
PtRu microkinetic modeling and inference are not yet implemented.

### 1. Process Todd measurements

```powershell
python scripts/process_ptru_todd.py
```

Processes Pt/PtRu measurements in HClO4 and KOH.

Outputs:

```text
data/processed/PtRu_COOx_acidic/
data/processed/PtRu_COOx_basic/
```

Each dataset contains standardized replicate observations and
full interpolated analysis observations.

### 2. Process Em measurements

```powershell
python scripts/process_ptru_em.py
```

Processes supplementary Pt/C data in acidic and basic media.

Em sample preparations are kept separate from Todd and from
one another. Pt/ATO measurements are currently excluded.

Em has one electrolyte concentration (0.10 M), so electrolyte
reaction orders cannot be calculated for these datasets.

### 3. Generate experimental rate figures

```powershell
python scripts/plot_ptru_todd.py
```

Generates linear-scale and logarithmic-scale TOF figures for
Todd and Em, including available replicate variability.

Outputs:

```text
figures/preprocessing/PtRu_COOx_acidic/
figures/preprocessing/PtRu_COOx_basic/
```

Todd figures are organized by material.
Em figures are organized by sample preparation.

All figures are saved as PNG and SVG.

### 4. Calculate experimental kinetic observables

```powershell
python scripts/process_ptru_observables.py
```

This consolidated script replaces the three earlier entry points:

```text
process_ptru_alpha.py
process_ptru_co_orders.py
process_ptru_ion_orders.py
```

It calculates:

- Apparent transfer coefficient (alpha), separately for each
  experimental replicate and then summarized.
- CO reaction order (delta_CO), using adjacent CO pressures.
- H+ reaction order (delta_Hplus), for acidic Todd measurements.
- OH- reaction order (delta_OH), for basic Todd measurements.

Electrolyte orders use one logarithmic slope across all four
electrolyte concentrations at matching SHE potentials.

PtRu reaction orders do not assume replicate pairing between
different experimental conditions. Available uncertainty bands
use propagated replicate variability.

The script writes observable Parquets under each dataset's
`analysis/` directory and generates the corresponding PNG/SVG
figures in the existing material or sample-preparation folders.

### Recommended complete PtRu workflow

Run from the repository root in this order:

```powershell
python scripts/process_ptru_todd.py
python scripts/process_ptru_em.py
python scripts/plot_ptru_todd.py
python scripts/process_ptru_observables.py
```

The first two commands produce standardized and interpolated
rate data. The last two generate figures and derived observables.

The PtRu analysis currently uses full, untruncated measurements.
Data-selection and truncation rules have not yet been finalized.

### PtRu implementation ownership

Reusable code:

```text
src/mkm/preprocessing/ptru_todd.py
src/mkm/preprocessing/ptru_em.py
src/mkm/preprocessing/ptru_observables.py
src/mkm/preprocessing/ptru_plotting.py
```

The CLI scripts handle orchestration rather than implementing
the experimental kinetic calculations or plotting algorithms.

---

## AgPd workflows

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

Independent per-material mechanism parameters with one shared error pair:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization independent --error-structure shared --overwrite
```

This fit samples 10 mechanism parameters independently for each alloy and the six active ER/LH/Pd parameters
for Pd100. The four Ag/BF-only Pd100 parameters (`deltaG5_0`, `beta_2_BF`, `q`, `Gact2_BF_0`) are not sampled.
The mechanism therefore has 56 free physical parameters plus the shared `sigma_rate_abs` and `sigma_rate_rel` pair.
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

Coverage-cap variants are registered but are currently intended for exploratory sensitivity analysis rather than the main model path. See `MODEL_REGISTRY.md` and the comments in `config/models/experimental/agpd_caps.yaml`.

## PowerShell chaining

Windows PowerShell 5 does not support `&&`. To run postprocessing followed by DRC only if postprocessing succeeds:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --plot-level full; if ($?) { python scripts/postprocess_agpd_drc.py CO_BF_ER --material Ag50Pd50 --check-half-step }
```

### All-material exclusions and one-command postprocessing

All-material fits can omit selected material datasets without changing the processed data files:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization independent --error-structure shared --exclude-materials Ag10Pd90 Ag90Pd10 --overwrite
```

The corresponding result stays in the usual all-material directory and uses a compact model-leaf suffix such as `CO_BF_ER_LH__no_Ag10Pd90_Ag90Pd10`. Pass the same `--exclude-materials` list when postprocessing. `postprocess_agpd_posterior.py` now runs the standard products, transition-state DRC products, and all-material composition-parameter products in one command by default. Use `--skip-drc` or `--skip-composition` only when needed.


## Pt100 / PtRu workflows (acidic and basic electrolyte)

Only **Pt100** is currently available for Bayesian fitting. Todd's PtRu-alloy data
and Em's Pt/C preparations are preprocessed independently; they must not be
silently included in a Pt100 fit. AgPd retains its own composition and
all-material model paths.

### Preprocessing and diagnostics

```powershell
python scripts/process_ptru_todd.py
python scripts/process_ptru_em.py
python scripts/process_ptru_observables.py --skip-plots
python scripts/plot_ptru_todd.py
python scripts/plot_ptru_observables.py
```

Todd's configured acidic Pt100 low-potential selection is applied before
fitting. Em preparations are not merged or truncated. Check source files,
concentration/activity conventions, and rate-normalization units before fitting.

### Fitting and postprocessing

```powershell
python scripts/check_pt100_prior_predictive.py CO_ER_LH --medium acidic --draws 1000
python scripts/fit_pt100_posterior.py CO_ER_LH --medium acidic
python scripts/postprocess_pt100_posterior.py CO_ER_LH --medium acidic --plot-level full
```

The prior-predictive check should be reviewed **before** starting an expensive
fit. The linear-rate Normal likelihood is shared with AgPd, but Pt100 presently
uses only a shared error structure. Postprocessing uses the shared 80% and 95%
HDI summaries, not the old Pt100 equal-tail quantile convention.

### Posterior checkpoint provenance

Post-P1 checkpoints include additional source hashes and registry metadata.
A posterior produced with the earlier implementation will ordinarily fail
strict validation with `model_specification` and `resolved_fit_sha256`
mismatches. Inspect such a checkpoint without changing it:

```powershell
python scripts/inspect_pt100_checkpoint.py CO_ER_LH --medium acidic
```

The audit distinguishes changes to inputs, prior specification, model
assumptions, source code, and derived fit hashes. A
`source_change_unverified` result means **the old and current implementations
have not been shown to be mathematically or numerically equivalent**. It is
not permission to override checkpoint validation, relabel metadata, resume
sampling, or interpret old posterior predictions under new code. Retain the
original checkpoint until independent equivalence checks are available.

### Comparison and validation

```powershell
python scripts/compare_pt100_models.py --medium acidic --comparison-name pathways --run LH,CO_LH --run ER,CO_ER --run ER_LH,CO_ER_LH
python scripts/validate_pt100_posterior.py CO_ER_LH --medium acidic --electrolyte-M 0.1 --co-mole-fraction 0.01
```

Comparison requires completed, compatible checkpoints with identical ordered
observations. Pointwise PSIS-LOO leaves out individual measured rates, **not**
correlated potential curves or whole electrolyte/CO conditions. Check Pareto-k
reliability, and do not interpret ELPD differences as mechanistic proof.
Condition-level LOCO refits all remaining conditions and excludes every
replicate and potential point from the held-out condition. Pt100 LOMO and
composition-dependent Bayesian fits are not implemented.

### Regression tests

The repository includes synthetic tests of Pt100 mechanism behavior, the
shared CO steady-state solver, Todd/Em preprocessing, postprocessing products,
checkpoint provenance, and Pt100 condition holdouts. They require no MCMC
sampling and should run alongside the full AgPd regression suite:

```powershell
pytest tests/test_shared_co_ssa.py tests/workflows/test_ptru_checkpoint_audit.py tests/workflows/test_ptru_postprocessing_smoke.py -q
pytest tests -q
```
