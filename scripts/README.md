# MKM workflow scripts

All commands below are intended to be run from the repository root with the `mkm` environment active.

The CLI scripts are deliberately thin. Reusable scientific/inference logic belongs under `src/mkm/`.

## PtRu preprocessing and observables

PtRu preprocessing covers Pt100, Pt66Ru33, and Pt50Ru50 in acidic
and basic media. Todd and Em measurements are processed separately.

The PtRu experimental pipeline produces rates, apparent transfer
coefficients, CO reaction orders, and H+/OH- reaction orders.
Individual and within-medium all-material microkinetic inference is
implemented for Pt100, Pt66Ru33, and Pt50Ru50, as described below.
Em preparations remain outside the fitted dataset.

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

Each dataset contains standardized replicate observations, full interpolated
analysis observations, selected observations, and truncation metadata.

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

Todd preprocessing preserves full observations and produces a separate
selected dataset. Configured low-potential exclusions currently affect only
Todd acidic Pt100; other Todd materials and Em data remain untruncated.
Experimental observable products are generated for both full and selected
Todd data. The selection thresholds remain scientific assumptions.

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

Individual AgPd and PtRu fits now default to `--error-structure shared`. Pooled fits
also default to shared, including `independent` physical parameterizations. Explicitly use
`--error-structure material` only for pooled fits that support it.

Individual fit identity:

```powershell
python scripts/check_agpd_prior_predictive.py CO_BF_ER --material Ag50Pd50
```

All-material linear-composition identity:

```powershell
python scripts/check_agpd_prior_predictive.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material
```

Use `--save-draws` only when the raw prior-predictive InferenceData is needed.

### Frumkin model numerical checks (AgPd)

Dynamic AgPd model names may begin with `c`, `x`, and/or `o` for CO-CO,
CO-OH, and OH-OH interactions. Finite-rate steps can also be selected in
the name. For example, run the fully coupled model at Ag50Pd50:

```powershell
python scripts/check_agpd_prior_predictive.py cxo_CO_BF_ER_LH_3_pOH_sOH --material Ag50Pd50 --draws 50 --random-seed 42 --save-draws
```

If the summary identifies nonfinite predictions, replay the saved parameter
draws with more iterations for diagnosis (this is not a model change):

```powershell
python scripts/diagnose_agpd_lateral_solver.py cxo_CO_BF_ER_LH_3_pOH_sOH --material Ag50Pd50 --iterations 384
```

The two-initial-state solver checks residuals and branch disagreement; it
cannot rule out all additional steady states. For independent root/stability
investigation use `scripts/diagnose_agpd_lateral_stability.py`. Solver
thresholds and iteration settings are defined by the model YAML, not the
CLI's numeric replay argument.

## Posterior fitting

### Individual material

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50
```

Common sampler override:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --target-accept 0.95 --overwrite
```

Individual AgPd fits use `--error-structure shared` (one error-parameter pair).

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

Independent per-material mechanism parameters with one shared error pair:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization independent --error-structure shared --overwrite
```

This fit samples 10 mechanism parameters independently for each alloy and the six active ER/LH/Pd parameters
for Pd100. The four Ag/BF-only Pd100 parameters (`deltaG5_0`, `beta_2_BF`, `q`, `Gact2_BF_0`) are not sampled.
The mechanism therefore has 56 free physical parameters plus the shared `sigma_rate_abs` and `sigma_rate_rel` pair.

LOMO requires shared likelihood errors. An `independent` physical parameterization also has no fitted mechanism-parameter vector for an omitted material, so it cannot extrapolate that material.

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
--skip-drc
--skip-composition   (all-material runs)
```

Ordinary AgPd postprocessing runs transition-state DRC by default and generates all-material composition products unless skipped. The standalone DRC command below can be used independently or for half-step diagnostics. When `--skip-drc` is used, previously computed DRC files and figures remain available but are **not recomputed**; see the `drc/drc_status.json` sidecar for the invocation status and known generation timestamp.

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

Coverage-cap variants are registered but are currently intended for exploratory sensitivity analysis rather than the main model path. See `src/mkm/models/agpd_registry.py`, `config/models/agpd_basic.yaml`, and the separate experimental settings file `config/models/experimental/agpd_caps.yaml`.

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


## Unified PtRu workflows (Pt100, Pt66Ru33, Pt50Ru50)

PtRu uses one mechanism registry, fit builder, and family of CLIs for all three Todd materials.
Both acidic and basic data have separate models, prior profiles, checkpoints, and output trees;
**never pool acid with base**. Em's Pt/C preparations are preprocessed separately and are not fitted here.

### Preprocessing

```powershell
python scripts/process_ptru_todd.py
python scripts/process_ptru_em.py
python scripts/process_ptru_observables.py --skip-plots
python scripts/plot_ptru_todd.py
python scripts/plot_ptru_observables.py
```

### Individual-material fitting (including pure Pt)

```powershell
python scripts/check_ptru_prior_predictive.py CO_ER_LH --medium acidic --material Pt100 --draws 1000
python scripts/fit_ptru_posterior.py CO_ER_LH --medium acidic --material Pt100 --overwrite
python scripts/postprocess_ptru_posterior.py CO_ER_LH --medium acidic --material Pt100 --plot-level full
python scripts/fit_ptru_posterior.py CO_BF_ER_LH --medium basic --material Pt66Ru33 --overwrite
python scripts/postprocess_ptru_posterior.py CO_BF_ER_LH --medium basic --material Pt66Ru33 --plot-level full
```

Pure Pt supports only `CO_LH`, `CO_ER`, and `CO_ER_LH`. Both alloys support all seven
BF/ER/LH pathway combinations. Ru OH adsorption transfers **one full electron**; BF coupling
is chemical, and there is **no fitted `q` or `beta_2_BF`**. Pt and Ru site balances are distinct,
with the nominal Ru fraction used as a provisional BF-neighbor availability factor. Rates are
normalized per Pt CO adsorption site.

### All-material fits within one electrolyte medium

```powershell
python scripts/fit_ptru_posterior.py CO_ER_LH --medium acidic --all-materials --parameterization shared --error-structure shared --overwrite
python scripts/fit_ptru_posterior.py CO_BF_ER_LH --medium basic --all-materials --parameterization linear_xRu --error-structure material --overwrite
python scripts/postprocess_ptru_posterior.py CO_BF_ER_LH --medium basic --all-materials --parameterization linear_xRu --error-structure material --plot-level full
python scripts/fit_ptru_posterior.py CO_BF_ER_LH --medium acidic --all-materials --parameterization independent --error-structure shared --overwrite
```

`shared` uses one physical parameter vector across materials. `independent` fits separate
physical parameters for each material, omitting BF-only parameters on Pt100 and requiring
shared likelihood errors. `linear_xRu` varies physical parameters with nominal Ru fraction.
The `material` error structure fits material-specific rate-normal noise scales. For a BF-only
model (`CO_BF`), exclude pure Pt using `--exclude-materials Pt100`; otherwise it would have
zero modeled CO oxidation rate. Optional material exclusions are supported for other pooled
fits as well. `--overwrite` removes the selected checkpoint folder; `--resume` and
`--overwrite` are mutually exclusive.

For both PtRu and AgPd, `--skip-drc` retains prior DRC tables and figures and records
the state in `drc/drc_status.json`; it does not imply that retained files reflect
the current invocation. When old DRC files lack a known generation timestamp,
`last_computed_at_utc` is null. Recompute DRC to update their provenance.

### Diagnostics and validation

```powershell
python scripts/postprocess_ptru_drc.py CO_ER_LH --medium acidic --material Pt100 --check-half-step
python scripts/compare_ptru_models.py --medium acidic --material Pt100 --comparison-name pathways --run LH,CO_LH --run ER,CO_ER --run ER_LH,CO_ER_LH
python scripts/inspect_ptru_checkpoint.py CO_ER_LH --medium acidic --material Pt100
python scripts/validate_ptru_posterior.py CO_ER_LH --medium acidic --material Pt100 --electrolyte-M 0.1 --co-mole-fraction 0.01
python scripts/validate_ptru_lomo.py CO_ER_LH --medium acidic --holdout-material Pt50Ru50 --parameterization linear_xRu
python scripts/plot_ptru_composition_parameters.py CO_ER_LH --medium basic --parameterization linear_xRu
python scripts/compare_ptru_models.py --medium basic --all-materials --comparison-name pooled --run shared,CO_ER_LH,shared,shared --run trend,CO_ER_LH,linear_xRu,material
```

Pass the same `--all-materials`, `--parameterization`, `--error-structure`, and
`--exclude-materials` arguments for diagnostic scripts operating on a pooled posterior.
LOCO validation supports one selected PtRu material; pooled PtRu LOMO is now available with
`shared` or `linear_xRu` physical parameterization and shared likelihood errors. The BF-only
mechanism is not valid for PtRu LOMO: a three-material dataset cannot leave one alloy out
while retaining two BF-capable training materials. New checkpoints use model-scoped source
and resolved-configuration provenance (schema 2). Legacy checkpoints retain their original
strict validation requirements and are not silently migrated. Pointwise PSIS-LOO
leaves out measured rates, not entire correlated curves or conditions.

### Regression tests

```powershell
pytest tests/test_pt100_mechanisms.py tests/test_ptru_alloy_mechanisms.py tests/test_ptru_model_registry.py -q
pytest tests/workflows/test_ptru_fit.py tests/workflows/test_ptru_alloy_fit.py tests/workflows/test_ptru_posterior.py -q
pytest tests -q
```

## Development and full regression tests

The complete regression command is:

```powershell
pytest tests -q --durations=30
```

For a smaller development run that excludes selected expensive and
integration-oriented tests:

```powershell
pytest tests -q -m "not slow and not sampling and not plotting and not integration"
```

The full suite passed **790 tests** on 19 November 2026 in approximately
**3 minutes**; the filtered subset took **2 minutes 30 seconds**. These
markers currently cover selected expensive tests, not all substantial
PyTensor compilations. The 790-pass baseline predates 13 new test cases
for zero-rate reporting and dynamic AgPd DRC consistency. To run those
tests alone:

```powershell
pytest tests/test_agpd_zero_rate_fractions.py tests/postprocessing/test_agpd_dynamic_drc.py -q
```

The DRC tests are marked `slow` and are omitted from the filtered
development subset. Keep the full suite as the merge/commit validation.

## Model registration, provenance and validation (November 2026)

The seven standard AgPd pathway combinations are selectable for all-material fits.
For pooled `CO_BF` fitting, use `--exclude-materials Pd100`; the
individual fitting workflow also retains its existing Pd100 BF-model
restrictions. These are *fitting* restrictions: direct dynamic `BF` and
`CO_BF` evaluation on Pd100 remains mathematically defined, giving
`ln_rate = -inf` and a reported BF fraction of zero. AgPd model
comparison and prior-predictive commands also accept `--exclude-materials`
for pooled fits. PtRu model comparison can use
`--run LABEL,MODEL,PARAMETERIZATION,ERROR_STRUCTURE` to compare different parameterizations
and likelihood error structures for an **identical ordered observation set**. LOCO and LOMO
hold out whole experimental groups; PSIS-LOO still holds out individual observations.

New posterior metadata use provenance schema 2. The source signature hashes the registered
mechanism entry and its statically reachable implementation rather than unrelated model
entries. Selected source signatures are not a guarantee of semantic equivalence. The
validator also hashes tracked shared kinetic and likelihood dependencies and compares resolved
kinetic conditions, priors, material set, fit parameterization, and data identity. Whole
configuration-file hashes remain in metadata for auditing, but do not invalidate schema-2
posteriors when unrelated model entries are added. This is a *static dependency signature*;
complex dynamic imports or monkeypatching require manual review. Version-1 checkpoints
are deliberately **not** accepted on the basis of approximate hashes; keep their original
Git revision/configuration available for provenance verification.
