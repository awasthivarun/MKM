# MKM Repository Review — 2026-08-23

## Review scope

This review covers the public `rebuild/from-scratch` branch and the uploaded local environment snapshots. Raw workbook and parquet contents were intentionally excluded; data files were reviewed only as named inputs/outputs and path contracts.

Validated local checkpoint supplied by the user:

- **192 tests passed**
- **7 known warnings**
- Python 3.13.15
- PyMC 6.3.1
- PyTensor 3.3.0
- NumPy 2.4.6
- SciPy 1.18.0
- Numba 0.66.0
- nutpie 0.16.11
- ArviZ split stack 1.3.0 (`arviz`, `arviz-base`, `arviz-plots`, `arviz-stats`)

## Executive conclusion

The rebuilt repository is scientifically coherent and has a strong reusable core. It does **not** need a rewrite.

The best parts are:

- nonrectangular `conditions` / `model_points` / `observations` architecture
- explicit replicate observations and setup indexing
- chemistry-independent PyMC model assembly
- separate mechanism equations, model registry, priors, and likelihood
- fixed linear observable maps for alpha and reaction orders
- persistent, tested postprocessing products
- validated transition-state DRC
- unusually strong test coverage for a research codebase

The main bottlenecks are now **workflow orchestration and metadata duplication**, not the mathematical core.

The highest-value refactors are:

1. centralize paths, configuration loading, material selection, and fit/postprocessing context
2. remove `Ag10Pd90` hard-coding from canonical scripts
3. split the 457-line posterior orchestrator into reusable stages
4. split the 634-line plotting module by scientific domain
5. move model-specific output/DRC metadata into a single model metadata contract
6. add fit provenance/run manifests and settle the generated-results policy
7. make the tested Conda environment reproducible

## What should remain stable

Do not broadly rewrite these components during housekeeping:

- `model_data.py`
- `observable_maps.py`
- mechanism equations
- the zero-sum setup likelihood mathematics
- paired CO-order preprocessing
- analytic CO-SSA solution
- transition-state DRC definition and current finite-difference validation

Those are functioning, physically motivated, and well tested.

# 1. Architecture review

## 1.1 Strong core: model-data boundary

The `ModelDataTables` design is the right abstraction:

- `conditions`
- `model_points`
- `observations`

It avoids a forced concentration × gas × potential tensor and supports condition-specific potential windows. This will matter when PtRu acid/basic datasets are introduced.

Recommendation: treat this as a stable internal contract. Add fields only when a new dataset genuinely requires them.

## 1.2 Strong core: observable maps

The linear-map representation

\[
q_k = \sum_j w_{kj}\ln r_j
\]

is one of the strongest design decisions in the repository. It provides a single definition for experimental/posterior alpha and reaction orders without rectangular indexing.

Recommendation: continue using this layer for any derived observable that is linear in model log rate. Do not reimplement these derivatives inside plotting code.

## 1.3 Strong core: mechanism / model / inference separation

The current separation is sensible:

- `mechanisms/`: equations and parameter/result dataclasses
- `models/`: registry + prior-profile binding
- `inference/`: PyMC assembly, likelihood, priors, sampling
- `postprocessing/`: posterior-derived calculations

This makes adding new models feasible without rewriting preprocessing or inference.

The limitation is that `AgPdModelDefinition` currently stores only:

- parameter dataclass
- evaluator

Other model-specific knowledge lives elsewhere.

# 2. Highest-priority organizational bottlenecks

## P1 — duplicated workflow context and path logic

### Current symptom

Canonical scripts independently define versions of:

- repository root
- processed-data paths
- preprocessing/model config paths
- posterior root
- output directory layout
- legacy-vs-current posterior fallback
- material filtering
- model-data construction
- setup-aware input construction

The same logic appears in fitting, posterior postprocessing, model comparison, DRC, prior predictive, and older diagnostic scripts.

### Why this matters

This is the biggest barrier to:

- fitting another material
- changing a results path
- removing legacy path fallback
- adding another dataset
- keeping scripts consistent

It also makes every script look more complicated than its scientific purpose.

### Recommended fix

Create a small, non-scientific workflow/context layer, for example:

```text
src/mkm/project_paths.py
src/mkm/workflows/agpd_basic.py
```

Possible immutable objects:

```python
@dataclass(frozen=True)
class AgPdPaths:
    root: Path
    analysis_root: Path
    selected_path: Path
    summary_path: Path
    delta_oh_path: Path
    delta_co_path: Path
    model_config_path: Path
    preprocessing_config_path: Path
    posterior_root: Path
```

```python
@dataclass(frozen=True)
class AgPdRunContext:
    material: str
    model_name: str
    likelihood_name: str
    model_config: dict
    preprocessing_config: dict
    model_data: ModelDataTables
    inputs: ModelInputArrays
    posterior_dir: Path
```

The context layer should load/validate paths and data. It should **not** hide mechanism equations or statistical calculations.

## P1 — material hard-coding

### Current symptom

The main scripts set:

```python
MATERIAL = "Ag10Pd90"
```

This includes fitting, posterior postprocessing, model comparison, and DRC.

### Why this matters

The repository already contains six AgPd materials. Once prior profiles are added, the scripts still cannot run them without source edits.

### Recommended fix

Add:

```text
--material Ag10Pd90
```

to every canonical model-facing script.

Validate the material against:

- processed data
- surface-composition config
- prior-profile config

Do not generalize to joint multi-material inference accidentally. A single-material CLI remains appropriate until a joint hierarchical model is explicitly designed.

## P1 — oversized posterior orchestrator

### Current symptom

`scripts/postprocess_agpd_posterior.py` is approximately 457 lines. Most heavy calculations have moved into modules, but the script still:

- loads configs/data/posterior
- builds inputs
- drives parameter diagnostics
- drives sampler diagnostics
- drives predictions/residuals
- computes LOO/PIT
- summarizes physical variables
- builds observable maps/comparisons
- saves all outputs
- plots all outputs
- prints all reports

### Why this matters

The script is now the main comprehension bottleneck. A missed keyword at this level previously broke the CLI despite passing unit tests.

### Recommended fix

Move orchestration into a package function returning a structured result:

```python
@dataclass(frozen=True)
class PosteriorPostprocessingResult:
    parameter_contraction: pd.DataFrame
    sampling: SamplingDiagnostics
    observation_diagnostics: pd.DataFrame
    residual_curves: pd.DataFrame
    shared_residuals: pd.DataFrame
    loo: LOOModelDiagnostics
    calibration: LOOCalibration
    physical_summary: pd.DataFrame
    balance_summary: pd.DataFrame
    pointwise: dict[str, pd.DataFrame]
    observable_comparisons: dict[str, ExperimentalObservableComparison]
```

Then use explicit stage functions:

```text
build context
calculate products
save products
plot products
print report
```

The CLI should eventually be approximately 50–100 lines.

## P1 — monolithic plotting module

### Current symptom

`src/mkm/postprocessing/plotting.py` is approximately 634 lines and contains unrelated figure families:

- parameter posterior plots
- sampling trace/rank/energy/pairs
- posterior predictions/residuals
- coverage/pathway plots
- experimental observable comparisons
- model comparison and pointwise ELPD
- Pareto-k
- LOO-PIT
- DRC

### Why this matters

Every new figure touches one large shared file. Plotting dependencies and domain-specific assumptions become hard to follow, and merge conflicts become likely.

### Recommended fix

Convert `plotting.py` into a package:

```text
src/mkm/postprocessing/plotting/
├── __init__.py
├── parameters.py
├── sampling.py
├── predictions.py
├── observables.py
├── loo.py
├── model_comparison.py
├── drc.py
└── common.py
```

`common.py` can own repeated condition-grid layout helpers.

Do not refactor plot aesthetics and file organization simultaneously. First move functions unchanged and preserve public imports through `plotting/__init__.py`.

## P1 — model-specific metadata is dispersed

### Current symptom

Knowledge about one model is spread across:

- `models/agpd_basic.py`: parameter dataclass + evaluator
- model YAML: prior parameter names
- `postprocessing/diagnostics.py`: hard-coded coverage/pathway names
- `postprocess_agpd_posterior.py`: `POINTWISE_VARIABLES`
- `postprocessing/drc.py`: transition-state controls
- plotting labels

### Why this matters

Adding `ER`, `BF_OH`, or another mechanism requires edits in several places beyond the mechanism and registry. The inference core is modular, but postprocessing is not fully registry-driven.

### Recommended fix

Extend the model definition or create a companion metadata object:

```python
@dataclass(frozen=True)
class AgPdModelMetadata:
    pointwise_variables: tuple[str, ...]
    pathway_fractions: tuple[str, ...]
    site_balances: dict[str, tuple[str, ...]]
    transition_state_controls: tuple[TransitionStateControl, ...]
```

This metadata should describe scientific outputs and checks, not plot styling.

A registry contract test should verify:

- every configured prior matches the parameter dataclass
- every DRC control names a mechanism parameter
- every declared pointwise variable is returned by the evaluator
- site-balance groups refer to returned coverage variables

## P1 — environment and package specification were absent

The old `pyproject.toml` contains no runtime dependencies, and the historical environment file lists `defaults`, while the actually resolved working environment is predominantly `conda-forge`.

This is a major reproducibility risk because PyMC/PyTensor/nutpie/ArviZ APIs have already proved version-sensitive.

Recommended policy:

- `pyproject.toml`: direct Python dependencies with bounded compatible ranges
- `environment.yml`: exact validated development stack
- `conda-explicit.txt`: optional Windows archival snapshot, not the primary environment definition
- install repository editable with `--no-deps` after creating the Conda environment

Use `conda-forge` with `nodefaults` for the validated environment to avoid silent channel mixing.

# 3. Results and provenance

## P1 — generated-results policy is unclear

The repository tracks selected CSV/Parquet/PNG products while `.gitignore` globally ignores `*.nc`.

Consequences:

- derived plots/tables may be versioned without the posterior needed to reproduce them
- NetCDF test fixtures would also be ignored
- result directories can grow rapidly as materials/models are added
- historical and canonical path schemes coexist

### Recommended policy

Preferred:

```text
results/   generated, normally ignored
figures/   generated, normally ignored
reports/   selected curated outputs committed
```

If generated outputs remain tracked, add `results/README.md` stating exactly what is committed and why.

Do not delete current historical results during this migration.

## P1 — fit provenance is incomplete

Every fit should write a `run_metadata.yaml` containing:

- git commit hash
- dataset/material/model/likelihood
- data-file hash
- config-file hashes
- sampler/backend
- chains/tune/draws/target acceptance
- random seed
- Python/package versions
- start/end timestamps
- hostname/platform if useful

This is more important than storing every resolved Conda transitive dependency beside every fit.

# 4. Statistical and scientific API observations

## Interval semantics

Several saved summaries use:

```text
q025 / q975
```

which are equal-tail quantiles. Parameter plots use a 95% HDI.

These are different uncertainty summaries.

Recommendation:

- retain `q025/q975` where equal-tail intervals are intended
- label them `ETI` in documentation/figures
- use `hdi95_lower/hdi95_upper` only when ArviZ HDI is actually computed
- do not describe all posterior bands generically as HDIs

## Likelihood naming

The CLI name `setup_intercept` actually selects a **zero-sum setup-intercept likelihood**.

Recommendation: either rename the canonical likelihood to:

```text
zero_sum_setup_intercept
```

or document that `setup_intercept` is currently defined to be zero-sum. Do not later add an unconstrained variant behind the same name.

## Dense centering matrix

The zero-sum likelihood constructs a dense setup-centering matrix and performs a dot product. It is correct and tiny for the current nine setups, but it scales as \(O(S^2)\) and caused the known Numba performance warning.

Recommendation: defer until multi-material work, then replace with grouped sums/indexing or a sparse/block representation. This is not a current correctness issue.

## LOO scope

The current observation-wise LOO is appropriate for:

> prediction of another observation from a setup otherwise represented in the fit

It is not leave-one-setup-out validation.

Recommendation: keep the current diagnostic and later add a grouped validation workflow when new-setup prediction becomes a scientific question.

## DRC

The transition-state DRC implementation is well conceived:

\[
X_{\mathrm{TS},j}=-k_BT\frac{\partial\ln r}{\partial G^\ddagger_j}
\]

Current validations are strong:

- BF: \(X_{\mathrm{BF}}=1\) to approximately \(10^{-13}\)
- BF_LH: sum rule to approximately \(10^{-13}\)
- CO-SSA composite: sum rule and half-step convergence to approximately \(10^{-7}\)

Current limitation:

- DRC loops over posterior draws and controls in Python
- each control requires plus/minus model evaluations
- acceptable now, likely a bottleneck across six materials/many models

Possible later optimization:

- compile a batched parameter evaluator
- use symbolic derivatives for transition-state parameters
- preserve finite differences as a validation reference

Do not implement intermediate thermodynamic-control coefficients until the held-fixed state/transition-state energy convention is explicit.

# 5. File-level review

## `src/mkm/mechanisms/agpd_basic.py`

Approximately 716 lines. It is large but still scientifically cohesive.

Recommendation: do not split only for line count. Split before the next major model family (e.g. OH-SSA/lateral interactions) into:

```text
mechanisms/agpd/
├── common.py
├── qea.py
├── co_ssa.py
├── evaluators.py
└── __init__.py
```

Re-export current public names to avoid a broad migration.

## `src/mkm/preprocessing/agpd_basic.py`

Approximately 623 lines with separable responsibilities:

- workbook parsing
- grid/interpolation
- rate normalization
- truncation
- summarization
- AgPd observable wrappers

Lower priority than workflow/postprocessing refactors because it is stable and well tested.

Potential eventual split:

```text
preprocessing/agpd_io.py
preprocessing/agpd_grid.py
preprocessing/agpd_observables.py
preprocessing/agpd_basic.py  # public façade
```

## `src/mkm/model_inputs.py`

The setup-indexing logic is clear and validates the experimental design. Preserve it.

Minor readability issue: some top-level function boundaries lack a blank line. Low priority.

## `src/mkm/postprocessing/__init__.py`

The current numerical-only export direction is good. Keep plotting imports explicit from `mkm.postprocessing.plotting`.

As the package grows, consider reducing root exports rather than exposing every helper. Add a test:

```python
for name in mkm.postprocessing.__all__:
    assert hasattr(mkm.postprocessing, name)
```

## Legacy scripts

The canonical workflow now supersedes several scripts:

- `compare_agpd_posterior_observables.py`
- `diagnose_agpd_posterior.py`
- `plot_agpd_posterior_geometry.py`
- parts of `diagnose_agpd_posterior_science.py`

Do not delete immediately. Mark as compatibility/development, verify no documentation references them, then archive or remove in a deliberate cleanup commit.

# 6. Testing review

## Strengths

The 192-test suite covers:

- preprocessing contracts
- model-data and model-input indexing
- observable maps
- mechanisms and limiting behavior
- priors/registry
- likelihood structure
- posterior sampling/reconstruction/log likelihood
- postprocessing calculations
- plotting smoke tests
- LOO/calibration
- transition-state DRC

This is unusually strong for a research repository.

## Gaps

Add:

1. CLI/orchestrator smoke tests with temporary output directories
2. package `__all__` integrity test
3. result schema tests for canonical postprocessing
4. one tiny end-to-end model → posterior-like fixture → postprocessing workflow
5. registered-model metadata completeness test
6. output provenance/run-manifest test
7. pytest markers for slow categories:
   - `sampling`
   - `integration`
   - `plotting`
   - `slow`

The earlier missed keyword in the canonical script is exactly the kind of failure a thin CLI integration test would catch.

# 7. Documentation review

## README

Current README is useful but stale and malformed near the command/script section.

Needs:

- environment setup
- fenced command examples
- canonical postprocessing/model-comparison/DRC commands
- current test checkpoint
- links to repo map/current state
- no script inventory appended at the end

## REPO_MAP

Current map predates the permanent postprocessing and DRC packages.

Needs:

- single-model postprocessing ownership
- multi-model comparison ownership
- DRC flow
- environment files
- current outputs
- current bottlenecks

## scripts/README

Current inventory omits the new canonical scripts and labels the old diagnostic path as current.

Needs explicit categories:

- canonical workflow
- compatibility
- diagnostic/development
- legacy/superseded

## CURRENT_STATE

Current file still describes postprocessing and DRC as future work and contains stale test counts.

It should be rewritten around the completed first-generation Ag10Pd90 workflow.

# 8. Recommended target architecture

```text
src/mkm/
├── project_paths.py
├── workflows/
│   └── agpd_basic.py
├── mechanisms/
├── models/
├── inference/
├── postprocessing/
│   ├── diagnostics.py
│   ├── predictions.py
│   ├── residuals.py
│   ├── observables.py
│   ├── observable_comparison.py
│   ├── loo.py
│   ├── calibration.py
│   ├── model_comparison.py
│   ├── drc.py
│   └── plotting/
│       ├── parameters.py
│       ├── sampling.py
│       ├── predictions.py
│       ├── observables.py
│       ├── loo.py
│       ├── model_comparison.py
│       ├── drc.py
│       └── common.py
└── ...
```

Canonical scripts should become thin wrappers around package-level workflows.

# 9. Prioritized implementation roadmap

## Phase 1 — reproducibility and low-risk organization

1. add updated `pyproject.toml`
2. add validated `environment.yml`
3. update documentation
4. add package API integrity test
5. add pytest markers
6. define generated-results policy
7. add run metadata to future fits

## Phase 2 — remove workflow duplication

1. add `project_paths.py`
2. add AgPd run-context loader
3. add `--material` to canonical scripts
4. remove repeated path/config/data-loading code
5. remove legacy path fallback only after migration

## Phase 3 — split orchestration and plotting

1. create package-level posterior workflow
2. reduce posterior script to CLI
3. split plotting by domain
4. add CLI integration tests

## Phase 4 — model extensibility

1. add model metadata contract
2. move DRC controls/pointwise variables/site balances into metadata
3. add metadata completeness tests
4. register additional existing mechanisms such as ER if scientifically desired

## Phase 5 — scientific expansion

1. prior sensitivity for legacy-tuned priors
2. other AgPd compositions
3. grouped/new-setup predictive validation
4. explicitly defined intermediate thermodynamic control
5. PtRu acid/basic
6. nonideal/lateral-interaction models only when intentionally prioritized

# Final assessment

The repository is in good scientific shape. Its bottlenecks are now typical of a successful research prototype that has grown into a workflow:

- too much repeated run context
- one oversized orchestrator
- one oversized plotting module
- model metadata spread across modules
- no reproducible environment/provenance contract

Fix those incrementally. Do not disturb the validated model-data, likelihood, mechanism, observable-map, or transition-state DRC mathematics while doing so.
