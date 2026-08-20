# MKM Rebuild Checklist

* [x] **Repository foundation**

  * [x] Create clean rebuild branch.
  * [x] Create clean Conda environment.
  * [x] Establish immutable `data/raw/` policy.
  * [x] Organize raw experimental datasets into dataset-specific directories.
  * [x] Add README files documenting raw-data structure and provenance.
  * [x] Create `data/processed/`.
  * [x] Create source layout:

    * [x] `src/mkm/`
    * [x] `src/mkm/__init__.py`
    * [x] `src/mkm/preprocessing/`
    * [x] `src/mkm/preprocessing/__init__.py`
  * [x] Create workflow layout:

    * [x] `scripts/`
    * [x] `config/preprocessing/`
    * [x] `tests/preprocessing/`
  * [x] Create minimal `pyproject.toml`.
  * [x] Install package in editable mode with `pip install -e .`.
  * [x] Install preprocessing/test dependencies.
  * [x] Confirm `mkm` imports correctly.
  * [x] Confirm pytest runs correctly.

* [x] **AgPd raw-data contract**

  * [x] Document materials:

    * [x] `Pd100`
    * [x] `Ag10Pd90`
    * [x] `Ag25Pd75`
    * [x] `Ag50Pd50`
    * [x] `Ag75Pd25`
    * [x] `Ag90Pd10`
  * [x] Document KOH concentrations:

    * [x] 0.25 M
    * [x] 0.50 M
    * [x] 1.00 M
  * [x] Document CO mole fractions:

    * [x] 0.001
    * [x] 0.01
    * [x] 0.10
    * [x] 1.00
  * [x] Document replicates A/B/C.
  * [x] Document SHE potential reference.
  * [x] Document AgPd-specific expectation of 1000 points.
  * [x] Document raw current representation as `ln(j)` where numerical `j` is in µA/cm².
  * [x] Encode this contract in `config/preprocessing/agpd_basic.yaml`.
  * [x] Avoid treating nominal potential bounds as hard validation constraints.
  * [x] Avoid assuming uniform potential spacing.

* [x] **Generic structural validation**

  * [x] Create `src/mkm/preprocessing/validation.py`.
  * [x] Define `DataValidationError`.
  * [x] Validate expected material identifiers.
  * [x] Validate expected experimental-condition values.
  * [x] Validate replicate identifiers.
  * [x] Validate individual curve arrays:

    * [x] 1D potential.
    * [x] 1D current.
    * [x] Equal potential/current lengths.
    * [x] Numeric values.
    * [x] Finite values.
    * [x] Optional dataset-specific expected point count.
  * [x] Do **not** require raw potential monotonicity.
  * [x] Do **not** require exact raw potential endpoints.
  * [x] Do **not** require identical replicate lengths generically.
  * [x] Add common-grid structural validation for processed replicate sets.
  * [x] Add standardized-dataframe validation.
  * [x] Use acquisition `point_index` rather than potential as the raw unique coordinate.

* [x] **AgPd Excel ingestion**

  * [x] Inspect an actual AgPd workbook.
  * [x] Confirm worksheet naming convention:

    * [x] `0.1% CO`
    * [x] `1% CO`
    * [x] `10% CO`
    * [x] `100% CO`
  * [x] Confirm potential column is `E (V vs SHE)`.
  * [x] Confirm current-column convention.
  * [x] Avoid regex parsing.
  * [x] Parse CO worksheet names using explicit string operations.
  * [x] Parse KOH concentration using explicit string operations.
  * [x] Parse replicate using explicit string operations.
  * [x] Validate all expected KOH × replicate combinations per worksheet.
  * [x] Validate all expected CO worksheets per workbook.
  * [x] Load one workbook into canonical long-form data.
  * [x] Load all six workbooks into one standardized dataframe.
  * [x] Reconstruct current density using:

    * [x] (j = \exp[\ln(j)])
  * [x] Preserve both `ln_j_uA_cm2` and `j_uA_cm2`.
  * [x] Preserve acquisition order with `point_index`.

* [x] **AgPd ingestion tests**

  * [x] Create `tests/preprocessing/test_agpd_basic.py`.
  * [x] Test CO worksheet parsing.
  * [x] Test current-column parsing.
  * [x] Test workbook loading.
  * [x] Test material/KOH/CO/replicate identities.
  * [x] Test expected number of curves.
  * [x] Test AgPd-specific 1000-point curves.
  * [x] Test `j = exp(ln_j)` reconstruction.
  * [x] Test standardized-row uniqueness.
  * [x] Run pytest successfully.
  * [x] All current tests pass.

* [x] **Standardized AgPd artifact**

  * [x] Create `scripts/process_agpd_basic.py`.
  * [x] Load YAML configuration from the repository.
  * [x] Read `data/raw/AgPd_COOx_basic/`.
  * [x] Run AgPd ingestion and validation.
  * [x] Create `data/processed/AgPd_COOx_basic/standardized/`.
  * [x] Save standardized dataset as Parquet.
  * [x] Install/use PyArrow as pandas' Parquet engine.
  * [x] Successfully process all six AgPd workbooks.
  * [x] Generate standardized 216,000-row AgPd dataset.

* [ ] **Replicate potential-grid processing — CURRENT TASK**

  * [ ] Add aligned-replicate processing to `agpd_basic.py`.
  * [ ] Group by actual experimental condition.
  * [ ] Detect whether replicate potential vectors are already identical.
  * [ ] Preserve identical native grids without interpolation.
  * [ ] Accept monotonic increasing sweeps.
  * [ ] Reverse monotonic decreasing sweeps into increasing model coordinates.
  * [ ] Reject non-monotonic sweeps at the model-grid stage rather than silently sorting them.
  * [ ] Require dataset-specific sweep selection before interpolation of non-monotonic data.
  * [ ] Never extrapolate outside a replicate's measured potential domain.
  * [ ] Interpolate `ln(j)` rather than linear `j`.
  * [ ] Reconstruct `j` after interpolation.
  * [ ] Distinguish `point_index` from processed `grid_index`.
  * [ ] Record whether each grid is `native` or `interpolated`.
  * [ ] Do not invent an interpolation grid when replicate grids differ.
  * [ ] Add explicit interpolation-grid configuration later where required.
  * [ ] Test aligned AgPd output.
  * [ ] Confirm all AgPd conditions retain native grids if they are identical.

* [ ] **General preprocessing architecture**

  * [ ] Define a canonical condition representation that does not assume a full concentration × pressure Cartesian product.
  * [ ] Introduce stable `condition_id`.
  * [ ] Define observation indexing independently of rectangular plotting layout.
  * [ ] Support arbitrary numbers of conditions.
  * [ ] Support arbitrary replicate count per condition.
  * [ ] Support missing pressure levels.
  * [ ] Support missing electrolyte concentrations.
  * [ ] Support materials with different experimental designs.
  * [ ] Support condition-specific potential windows.
  * [ ] Support replicate-specific raw potential windows.
  * [ ] Separate:

    * [ ] raw acquisition coordinate
    * [ ] selected physical sweep
    * [ ] aligned replicate grid
    * [ ] model fitting grid
  * [ ] Decide whether reusable potential-grid utilities should live in a generic preprocessing module rather than dataset-specific modules.
  * [ ] Add provenance metadata for every transformation.

* [ ] **Potential-reference normalization**

  * [ ] Define canonical internal potential reference for CO oxidation.
  * [ ] Likely use SHE internally for current CO-oxidation datasets.
  * [ ] Preserve original reported reference.
  * [ ] Implement RHE → SHE conversion for PtRu datasets.
  * [ ] Verify concentration/pH dependence of the conversion.
  * [ ] Preserve both supplied and converted potentials where useful.
  * [ ] Test conversion numerically.
  * [ ] Ensure condition-dependent converted potential bounds are preserved.
  * [ ] Do not force converted PtRu conditions onto artificial global bounds.

* [ ] **Current-density → rate / TOF normalization**

  * [ ] Locate/document AgPd normalization information.
  * [ ] Define physical current-density → molar-rate conversion.
  * [ ] Define electron stoichiometry used in the conversion.
  * [ ] Define electrochemically active-area/site normalization.
  * [ ] Define TOF reference state.
  * [ ] Check dimensions explicitly.
  * [ ] Implement conversion as reusable scientific code.
  * [ ] Avoid embedding normalization constants as unexplained magic numbers.
  * [ ] Preserve current density alongside derived rate.
  * [ ] Add unit tests for normalization.

* [ ] **Replicate statistics**

  * [ ] Calculate replicate-resolved rate/TOF.
  * [ ] Calculate arithmetic mean rate.
  * [ ] Calculate rate SD.
  * [ ] Calculate replicate-resolved log-rate.
  * [ ] Calculate mean log-rate.
  * [ ] Calculate log-rate SD.
  * [ ] Decide treatment of single-replicate conditions.
  * [ ] Do not fabricate experimental uncertainty when only one replicate exists.
  * [ ] Decide how single-replicate uncertainty is represented downstream.
  * [ ] Test variable replicate counts.

* [ ] **Experimental transfer coefficient**

  * [ ] Define experimental (\alpha) convention.
  * [ ] Maintain consistency with model-side definition:

    * [ ] (\alpha=(RT/F)\partial\ln r/\partial E)
  * [ ] Decide differentiation method.
  * [ ] Decide whether smoothing is needed.
  * [ ] Make smoothing/differentiation settings explicit.
  * [ ] Avoid differentiating across nonphysical sweep reversals.
  * [ ] Calculate alpha per replicate where possible.
  * [ ] Calculate alpha mean/uncertainty across replicates.
  * [ ] Validate on synthetic functions with known derivative.

* [ ] **Experimental electrolyte reaction order**

  * [ ] Build generic concentration-order machinery.
  * [ ] For basic CO oxidation expose result as `delta_OH`.
  * [ ] For acidic CO oxidation expose result as `delta_H`.
  * [ ] Do not make plotting code intrinsically depend on either name.
  * [ ] Calculate only where sufficient concentrations exist.
  * [ ] Handle incomplete concentration sets.
  * [ ] Determine common potential domains across contributing conditions.
  * [ ] Decide interpolation strategy for cross-concentration comparison.
  * [ ] Define uncertainty propagation from replicate data.
  * [ ] Do not assume replicate A at one concentration is paired with replicate A at another unless experimentally justified.
  * [ ] Validate on synthetic (r\propto C^n) data.

* [ ] **Experimental CO reaction order**

  * [ ] Define local adjacent-pressure order convention.
  * [ ] Calculate only where the required pressure pair actually exists.
  * [ ] Do not assume every material has every pressure.
  * [ ] Determine common potential domain for each pressure pair.
  * [ ] Handle unequal replicate counts.
  * [ ] Define uncertainty propagation.
  * [ ] Preserve exact pressure/composition pair metadata.
  * [ ] Validate on synthetic (r\propto P_\mathrm{CO}^n) data.

* [ ] **Truncation / fitting-window logic**

  * [ ] Separate full processed curves from model-fitting curves.
  * [ ] Define dataset-specific fitting-window rules.
  * [ ] Permit different `(condition)` windows.
  * [ ] Permit different materials to use different windows where scientifically required.
  * [ ] Never overwrite standardized data.
  * [ ] Record reason for each truncation.
  * [ ] Record original and retained potential bounds.
  * [ ] Test truncation logic.
  * [ ] Review whether alpha-minimum truncation remains scientifically appropriate.

* [ ] **Model-ready dataset contract**

  * [ ] Replace dependence on a full `(C, P)` Cartesian dictionary.
  * [ ] Define canonical model input arrays.
  * [ ] Define condition-to-observation index mapping.
  * [ ] Preserve replicate dimension for the likelihood.
  * [ ] Support ragged experimental designs before final flattening.
  * [ ] Define flattened arrays only at model boundary.
  * [ ] Preserve metadata needed to reverse flattened predictions back to conditions.
  * [ ] Define full-range plotting data.
  * [ ] Define fitting-range data.
  * [ ] Define experimental observable storage.
  * [ ] Save model-ready artifacts reproducibly.
  * [ ] Add processing metadata and source hashes.
  * [ ] Add model-ready validation.

* [ ] **PtRu basic-data preprocessing**

  * [ ] Create PtRu-basic YAML contract describing actual available data.
  * [ ] Do not require complete pressure/concentration combinations.
  * [ ] Support variable replicate counts.
  * [ ] Support replicate-specific potential arrays.
  * [ ] Convert RHE → SHE.
  * [ ] Select physically relevant sweeps where raw potential is non-monotonic.
  * [ ] Align replicates onto condition-specific common grids.
  * [ ] Preserve varying condition potential bounds.
  * [ ] Convert current to rate/TOF.
  * [ ] Calculate supported kinetic observables only where sufficient data exist.
  * [ ] Generate canonical model-ready representation.
  * [ ] Add tests.

* [ ] **PtRu acid-data preprocessing**

  * [ ] Create acid-specific raw-data contract.
  * [ ] Support incomplete experimental matrix.
  * [ ] Handle potential-reference conversion.
  * [ ] Align variable replicate grids.
  * [ ] Calculate `delta_H` rather than `delta_OH`.
  * [ ] Keep generic concentration-order internals reaction/environment agnostic.
  * [ ] Generate canonical model-ready representation.
  * [ ] Add tests.

* [ ] **Model-input infrastructure**

  * [ ] Redesign current `process_experimental_data()` concept.
  * [ ] Remove assumption that every concentration × pressure combination exists.
  * [ ] Build model inputs from actual condition records.
  * [ ] Generate:

    * [ ] `E_in`
    * [ ] electrolyte concentration input
    * [ ] CO pressure/composition input
    * [ ] material metadata if needed
    * [ ] replicate observations
  * [ ] Preserve index maps for later reconstruction.
  * [ ] Keep calculations in log space where practical.
  * [ ] Add model-input validation.
  * [ ] Test missing-condition designs.

* [ ] **Reaction/model abstraction**

  * [ ] Separate electrochemical-reaction definition from generic inference code.
  * [ ] Define CO-oxidation-specific inputs and observables.
  * [ ] Avoid embedding `CO`, `OH`, or `H` assumptions into generic plotting infrastructure.
  * [ ] Keep potential-dependence implementation inside reaction/mechanism code.
  * [ ] Prepare architecture for future non-CO reactions.
  * [ ] Prepare architecture for reduction reactions.
  * [ ] Prepare architecture for HER/HOR.
  * [ ] Permit Butler–Volmer behavior near equilibrium.
  * [ ] Do not assume a Tafel approximation globally.
  * [ ] Preserve equilibrium-potential/reference-state information when required.

* [ ] **PyMC model infrastructure**

  * [ ] Rebuild base model interface.
  * [ ] Define physical parameter domains.
  * [ ] Define thermodynamic conventions.
  * [ ] Define coverage/site-balance constraints.
  * [ ] Define potential dependence consistently.
  * [ ] Define concentration/pressure dependence consistently.
  * [ ] Keep symbolic differentiable PyTensor expressions where practical.
  * [ ] Define likelihood for replicate-resolved observations.
  * [ ] Handle heteroscedastic error correctly.
  * [ ] Add prior-predictive checks.
  * [ ] Add synthetic recovery tests.
  * [ ] Check structural identifiability.
  * [ ] Check practical identifiability.

* [ ] **Posterior derived-observable infrastructure**

  * [ ] Rebuild model-side alpha calculation.
  * [ ] Rebuild concentration-order calculation using actual available conditions.
  * [ ] Rebuild CO-order calculation using actual available pressure pairs.
  * [ ] Build residuals from stable observation indexing.
  * [ ] Preserve condition metadata in posterior derived quantities.
  * [ ] Avoid assuming equal condition lengths.
  * [ ] Add surface-coverages as properly indexed deterministics/postprocessing variables.
  * [ ] Add DRC infrastructure.
  * [ ] Verify DRC definitions for activation energies, resistances, and other parameter types.
  * [ ] Check DRC sum behavior only where mathematically expected.

* [ ] **Inference diagnostics**

  * [ ] Posterior summaries.
  * [ ] Prior-to-posterior contraction.
  * [ ] Pair plots where useful.
  * [ ] Divergence diagnostics.
  * [ ] Energy/BFMI diagnostics.
  * [ ] R-hat.
  * [ ] ESS.
  * [ ] LOO.
  * [ ] Pareto-k.
  * [ ] Pointwise LOO.
  * [ ] LOO-PIT.
  * [ ] Posterior predictive checks.
  * [ ] Residual structure.
  * [ ] Qualitative R² only where clearly labeled as such.
  * [ ] Separate statistical adequacy from mechanistic support.

* [ ] **Unified plotting architecture**

  * [ ] Remove plotting dependence on a rectangular `n_C × n_P` dataset.
  * [ ] Plot only conditions that actually exist.
  * [ ] Define generic observable plotting interface.
  * [ ] Support experimental mean/uncertainty where available.
  * [ ] Support posterior mean/median and HDIs.
  * [ ] Support posterior-predictive HDIs where appropriate.
  * [ ] Rate plots.
  * [ ] Log-rate plots.
  * [ ] Log-residual plots.
  * [ ] Pointwise LOO plots.
  * [ ] Transfer-coefficient plots.
  * [ ] Concentration-order plots.
  * [ ] CO-order plots.
  * [ ] Surface-coverage plots.
  * [ ] DRC plots.
  * [ ] Handle missing conditions gracefully.
  * [ ] Handle unequal potential windows.
  * [ ] Handle different materials with different condition sets.
  * [ ] Allow reaction-specific labels without reaction-specific plotting algorithms.
  * [ ] Preserve consolidated plotting options where scientifically useful.

* [ ] **Model comparison**

  * [ ] Define common observation sets before comparing models.
  * [ ] Ensure LOO comparisons use compatible data.
  * [ ] Compare mechanistic variants.
  * [ ] Inspect condition-specific failure regions.
  * [ ] Compare posterior predictive behavior.
  * [ ] Compare parameter identifiability.
  * [ ] Compare physical plausibility.
  * [ ] Compare coverages and DRCs.
  * [ ] Do not treat favorable LOO/WAIC as proof of mechanism.

* [ ] **Scientific validation**

  * [ ] Unit consistency throughout.
  * [ ] Reference-state consistency.
  * [ ] Thermodynamic consistency.
  * [ ] Coverage bounds.
  * [ ] Site balance.
  * [ ] Chemically possible states.
  * [ ] Correct potential dependence.
  * [ ] Correct concentration dependence.
  * [ ] Correct pressure dependence.
  * [ ] Numerical stability.
  * [ ] Differentiability.
  * [ ] Sampler geometry.
  * [ ] Prior predictive plausibility.
  * [ ] Posterior predictive plausibility.
  * [ ] Residual diagnostics.
  * [ ] Identifiability.
  * [ ] Mechanistic observables.
  * [ ] Multiple rate-controlling steps where supported.
  * [ ] Electronic vs bifunctional interpretations only where data support them.

* [ ] **Repository reproducibility/documentation**

  * [ ] Create/update `DATA_CONTRACT.md`.
  * [ ] Create/update `CONVENTIONS.md`.
  * [ ] Create/update `REPO_MAP.md`.
  * [ ] Create/update `MODEL_REGISTRY.md`.
  * [ ] Create `CURRENT_STATE.md`.
  * [ ] Put this checklist or a condensed version in `CURRENT_STATE.md`.
  * [ ] Keep dataset READMEs synchronized with actual raw contracts.
  * [ ] Document preprocessing configuration.
  * [ ] Document generated processed artifacts.
  * [ ] Decide which processed artifacts belong in Git versus regeneration-only.
  * [ ] Add appropriate `.gitignore` entries.
  * [ ] Record dependency environment reproducibly.
  * [ ] Keep tests runnable from repo root.
  * [ ] Require preprocessing to be reproducible from immutable raw data.
