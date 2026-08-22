# scripts/README.md

Script inventory and current status.

## Canonical workflow scripts

- `process_agpd_basic.py` — **canonical**
  - Ingests raw AgPd workbooks and writes standardized/analysis/derived observable parquet outputs.

- `plot_agpd_basic.py` — **canonical**
  - Generates preprocessing summary plots from processed AgPd tables.

- `fit_agpd_posterior.py` — **canonical**
  - Fits configured AgPd model posteriors (iid or setup-intercept likelihood) and writes posterior artifacts/diagnostics.

## Current diagnostic scripts

- `check_agpd_prior_predictive.py` — **diagnostic**
  - Runs prior predictive checks and writes prior-derived summaries.

- `diagnose_agpd_likelihood.py` — **diagnostic**
  - Summarizes replicate dispersion/centered residual structure from processed data.

- `diagnose_agpd_setup_structure.py` — **diagnostic**
  - Quantifies replicate/setup offset structure in processed observations.

- `compare_agpd_posterior_observables.py` — **diagnostic**
  - Compares posterior derived observables with experimental summaries and writes comparison outputs/plots.

- `diagnose_agpd_posterior_science.py` — **diagnostic**
  - Produces broad posterior science diagnostics (parameter contraction, residual structure, physical checks, plots).

## Development / exploratory / legacy scripts

- `smoke_agpd_posterior.py` — **development**
  - Short smoke-run posterior check for fast environment/model sanity.

- `plot_agpd_posterior_geometry.py` — **development**
  - Ad hoc trace/pair plotting focused on posterior geometry inspection.

- `diagnose_agpd_posterior.py` — **legacy/superseded**
  - Earlier posterior diagnostic entry point that assumes older output layout (`.../posterior/<material>/<model>/...`) and is superseded by likelihood-aware scripts.

## Path-convention note

Posterior scripts currently support historical path variants in some places, including:

- `results/AgPd_COOx_basic/posterior/Ag10Pd90/<model>` (legacy)
- `results/AgPd_COOx_basic/posterior/Ag10Pd90/<likelihood>/<model>` (current direction)

Do not delete or move historical outputs during housekeeping; centralization is a future migration task.
