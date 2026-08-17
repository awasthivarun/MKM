# CO Oxidation Data Contract

## Purpose

This document defines the processed experimental-data structures used by the CO oxidation microkinetic models.

The data contract describes:
- variable meanings,
- units,
- array shapes,
- ordering conventions,
- experimental-condition keys,
- distinction between fitted and plotting-only data.

---

## Experimental dimensions

Each individual-material experiment varies:

- `C` — electrolyte concentration
  - base: `C_KOH`, units M
  - acid: `C_H`, units M
- `P_CO` — CO partial pressure, units atm
- `E` — electrode potential, units V vs SHE
- replicate/trial number

Current individual-material condition key:

`(C, P_CO)`

Target multi-material condition key:

`(material, C, P_CO)`

---

## Processed experimental dictionary

For an individual material:

`experiments[(C, P_CO)]`

contains the following fields.

### Model-input data

#### `truncated_E`
Potential values used for model fitting.

- units: V vs SHE
- shape: `(n_E,)`

#### `truncated_rate`
Mean experimental TOF across replicates.

- units: s^-1
- shape: `(n_E,)`

#### `truncated_rate_SD`
Standard deviation of experimental TOF across replicates.

- units: s^-1
- shape: `(n_E,)`

#### `truncated_log_rate`
Mean log experimental rate.

- quantity: natural logarithm of TOF
- shape: `(n_E,)`

#### `truncated_log_rate_SD`
Standard deviation of log experimental rate across replicates.

- shape: `(n_E,)`

#### `truncated_rate_matrix`
Individual replicate rates used as the PyMC observed variable.

Expected conceptual shape:

`(n_trials, n_E)`

The exact stored orientation must be verified and preserved by automated tests before refactoring.

---

## Full-grid plotting data

#### `E`
Full potential grid used for plotting rate, log-rate, and alpha.

- units: V vs SHE

#### `rate`
Mean experimental TOF on the full plotting grid.

- units: s^-1

#### `rate_SD`
Experimental TOF standard deviation.

- units: s^-1

#### `log_rate`
Mean natural log TOF.

#### `log_rate_SD`
Standard deviation of natural log TOF.

#### `alpha`
Experimental transfer coefficient.

#### `alpha_SD`
Experimental transfer-coefficient standard deviation.

---

## Electrolyte reaction-order data

Base:

- `E_OH`
- `delta_OH`
- `delta_OH_SD`

Acid:

- `E_H`
- `delta_H`
- `delta_H_SD`

The electrolyte reaction order is unique for each `P_CO`.

The same values may currently appear redundantly under multiple concentration keys in the processed dictionary.

---

## CO reaction-order data

- `E_CO`
- `delta_CO`
- `delta_CO_SD`

`delta_CO` is calculated between consecutive CO pressures.

Therefore, for `n_P` CO pressures, there are:

`n_P - 1`

CO reaction-order comparisons.

The highest CO pressure does not have a corresponding forward `delta_CO`.

---

## Flattened model-input arrays

The backend constructs:

- `E_in`
- `C_H_in`
- `C_KOH_in`
- `P_CO_in`
- `rate_obs`
- `rate_SD_obs`
- `log_rate_obs`
- `log_rate_SD_obs`
- `rate_obs_matrix`

All condition-dependent one-dimensional model-input vectors must share the same flattened observation ordering.

### Current individual-material ordering

Current processing loops in the order:

1. concentration
2. CO pressure
3. potential within each condition

Conceptually:

`(C1,P1) -> (C1,P2) -> ... -> (C2,P1) -> ...`

This ordering must be tested before and after refactoring.

---

## Target mega-model representation

Every observation must carry:

- material identity
- material composition
- electrolyte concentration
- CO pressure
- potential

For AgPd, the minimum metadata are:

- `material`
- `x_Ag`
- `E`
- `C_KOH`
- `P_CO`

Nominal compositions:

- Ag10Pd90: `x_Ag = 0.10`
- Ag25Pd75: `x_Ag = 0.25`
- Ag50Pd50: `x_Ag = 0.50`
- Ag75Pd25: `x_Ag = 0.75`
- Ag90Pd10: `x_Ag = 0.90`
- Pd100: `x_Ag = 0.00`

Nominal Ag composition should not automatically be interpreted as surface Ag fraction unless independently justified.

---

## Target indexing convention

The refactored backend should create one authoritative mapping:

`condition_slices[(material, C, P_CO)] -> slice(start, stop)`

All downstream calculations must use this mapping rather than independently reconstructing flattened-array indices.

This includes:

- posterior residuals
- alpha
- electrolyte reaction orders
- CO reaction orders
- model-fit plots
- coverage plots
- LOO plots
- DRC calculations

---

## Replicates and likelihood

The current PyMC likelihood uses individual replicate rates from:

`rate_obs_matrix`

Experimental log-rate uncertainty is available from:

`log_rate_SD_obs`

The exact relationship between replicate observations and experimentally estimated standard deviations must be verified before modifying the likelihood/error model.

No assumption should be made that zero SD means zero measurement uncertainty; zero SD may indicate that only one experimental replicate was available.

---

## Numerical conventions

- Natural logarithms are used for log-rate quantities.
- Model calculations should remain in log space whenever practical.
- Concentrations used inside logarithms must be dimensionless relative to the chosen standard state.
- CO pressures used inside logarithms must be dimensionless relative to the chosen standard state.
- Energies are generally expressed in eV.
- Temperature is currently fixed globally by the backend configuration.