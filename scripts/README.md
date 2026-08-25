# Script inventory

Scripts are command-line entry points. Reusable calculations belong in `src/mkm/`.

Use likelihood and composition arguments explicitly in scientific runs rather than relying on defaults.

## Data workflow

### `process_agpd_basic.py`

Builds standardized and selected AgPd analysis tables, rates, alpha, OH order, and paired CO order.

```powershell
python scripts/process_agpd_basic.py
```

### `plot_agpd_basic.py`

Generates preprocessing/data-summary figures from processed tables.

```powershell
python scripts/plot_agpd_basic.py
```

## Individual-material inference

### `check_agpd_prior_predictive.py`

Runs the current individual-material prior predictive checks and saves numerical summaries.

```powershell
python scripts/check_agpd_prior_predictive.py
```

Current limitation: this is not yet a composition-aware `linear_xAg` prior-predictive workflow.

### `fit_agpd_posterior.py`

Fits one registered AgPd mechanism to one material.

```powershell
python scripts/fit_agpd_posterior.py BF_LH --likelihood iid
```

### `postprocess_agpd_posterior.py`

Canonical individual-material posterior workflow. Generates:

- sampler diagnostics
- parameter diagnostics
- predictions and residuals
- coverages/pathways
- experimental observable comparisons
- pointwise PSIS-LOO/Pareto-k
- LOO-PIT calibration

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood iid
```

### `compare_agpd_models.py`

Individual-material multi-model comparison.

```powershell
python scripts/compare_agpd_models.py --likelihood iid
```

### `postprocess_agpd_drc.py`

Computes posterior transition-state DRC, sum-rule checks, finite-difference convergence, and DRC plots.

```powershell
python scripts/postprocess_agpd_drc.py BF_LH --likelihood iid --check-half-step
```

Current limitation: DRC is not yet composition-specific.

## Composition inference

### `fit_agpd_composition_posterior.py`

Fits a multi-material AgPd composition model.

Shared energetics:

```powershell
python scripts/fit_agpd_composition_posterior.py CO_BF_ER_LH --composition-model shared --likelihood iid
```

Linear Ag-composition energetics:

```powershell
python scripts/fit_agpd_composition_posterior.py CO_BF_ER_LH --composition-model linear_xAg --likelihood iid
```

For `linear_xAg`, configured parameters follow

\[
p(x_{\mathrm{Ag}})=p_{0.5}+s_p(x_{\mathrm{Ag}}-0.5).
\]

### `postprocess_agpd_composition_posterior.py`

Canonical multi-material posterior workflow.

```powershell
python scripts/postprocess_agpd_composition_posterior.py CO_BF_ER_LH --composition-model linear_xAg --likelihood iid
```

Persistent outputs include:

- sampler diagnostics
- posterior reference-parameter and slope summaries
- material-resolved noise and residual summaries
- rate/residual figures by material
- coverages/pathway fractions
- experimental alpha/OH-order/CO-order comparisons
- pointwise PSIS-LOO and Pareto-k
- LOO-PIT diagnostics

### `compare_agpd_composition_models.py`

Compares composition fits using saved posterior/log-likelihood products.

Current limitation: the CLI is oriented toward comparing mechanisms within the same composition parameterization. Direct comparison of

```text
shared / CO_BF_ER_LH
```

against

```text
linear_xAg / CO_BF_ER_LH
```

still needs a small CLI change. Do not rely on an old likelihood default; specify `--likelihood iid` explicitly.

## Predictive validation to add

### LOCO

Leave one material-specific `(KOH, CO)` condition out, removing all three replicate curves A/B/C for that condition.

Purpose: prediction of an unseen experimental condition within a known material.

### LOMO

Leave one material/composition out entirely.

Purpose: prediction of an unseen alloy composition and direct validation of the `linear_xAg` assumption.

Observation-wise PSIS-LOO remains useful for locating pointwise predictive problems but is not a substitute for LOCO or LOMO.

## Compatibility / diagnostic scripts

### `diagnose_agpd_posterior_science.py`

Compatibility wrapper forwarding to `postprocess_agpd_posterior.py`.

New documentation should use the canonical postprocessing command.

### `diagnose_agpd_likelihood.py`

Explores experimental replicate dispersion and likelihood assumptions.

### `diagnose_agpd_setup_structure.py`

Quantifies persistent A/B/C setup structure across paired CO series.

These are experimental-design diagnostics, not replacements for posterior postprocessing.

## Development / exploratory / superseded

### `smoke_agpd_posterior.py`

Short posterior sampling smoke/benchmark run.

### `plot_agpd_posterior_geometry.py`

Ad hoc posterior geometry plots. Superseded for routine use by canonical sampler/pair plots.

### `compare_agpd_posterior_observables.py`

Earlier standalone posterior-observable comparison. Superseded by canonical postprocessing.

### `diagnose_agpd_posterior.py`

Older posterior diagnostic using legacy output paths. Superseded.

## Path convention

Individual-material posterior:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
```

Composition posterior:

```text
results/AgPd_COOx_basic/posterior/composition/<composition_model>/<likelihood>/<model>/
```

New outputs should use the canonical likelihood-aware hierarchy.
