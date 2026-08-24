# Script inventory

Scripts are command-line entry points. Reusable calculations belong in `src/mkm/`.

## Canonical workflow

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

### `check_agpd_prior_predictive.py`

Runs configured Ag10Pd90 prior predictive checks and saves numerical summaries.

```powershell
python scripts/check_agpd_prior_predictive.py
```

### `fit_agpd_posterior.py`

Fits one registered AgPd model with the selected likelihood.

```powershell
python scripts/fit_agpd_posterior.py BF_LH --likelihood setup_intercept
```

### `postprocess_agpd_posterior.py`

Canonical single-model posterior workflow. Generates:

- sampler diagnostics
- parameter diagnostics
- predictions and residuals
- coverages/pathways
- experimental observable comparisons
- pointwise PSIS-LOO/Pareto-k
- LOO-PIT calibration

```powershell
python scripts/postprocess_agpd_posterior.py BF_LH --likelihood setup_intercept
```

### `compare_agpd_models.py`

Canonical multi-model comparison. Contains only products requiring multiple fits:

- stacking/model-comparison table
- aggregate ELPD differences
- pointwise ELPD differences

```powershell
python scripts/compare_agpd_models.py --likelihood setup_intercept
```

### `postprocess_agpd_drc.py`

Computes posterior transition-state DRC, sum-rule checks, finite-difference convergence, and DRC plots.

```powershell
python scripts/postprocess_agpd_drc.py BF_LH --likelihood setup_intercept --check-half-step
```

## Compatibility entry point

### `diagnose_agpd_posterior_science.py`

Compatibility wrapper forwarding to `postprocess_agpd_posterior.py`.

New documentation should use the canonical postprocessing command.

## Focused diagnostic scripts

### `diagnose_agpd_likelihood.py`

Explores experimental replicate dispersion and preliminary likelihood assumptions.

### `diagnose_agpd_setup_structure.py`

Quantifies persistent A/B/C setup structure across paired CO series.

These are experimental-design diagnostics, not replacements for posterior postprocessing.

## Development / exploratory / superseded

### `smoke_agpd_posterior.py`

Short posterior sampling smoke/benchmark run. Keep as a development tool and make model/likelihood arguments explicit before broader use.

### `plot_agpd_posterior_geometry.py`

Ad hoc posterior geometry plots. Superseded for routine use by the ArviZ sampler/pair plots in canonical postprocessing.

### `compare_agpd_posterior_observables.py`

Earlier standalone posterior-observable comparison. Superseded by canonical posterior postprocessing.

### `diagnose_agpd_posterior.py`

Older posterior diagnostic using legacy output paths. Superseded.

## Path convention

Canonical posterior path:

```text
results/AgPd_COOx_basic/posterior/<material>/<likelihood>/<model>/
```

Historical paths without the likelihood level remain for provenance. Do not create new outputs under the legacy hierarchy.

## Current limitation

Canonical scripts still hard-code `Ag10Pd90`. The next workflow refactor should add `--material` and centralize path/config/context construction.
