# PtRu CO Oxidation — Acidic Media

## Contents
This directory contains raw electrochemical CO oxidation data measured in acidic media.
Two Excel workbooks are included:
* `PtRu_acidic_current_densities_Todd.xlsx`
* `Pt100_PtATO_100mM_acidic_current_densities_Em.xlsx`

The two workbooks contain different catalyst sets and experimental condition grids and should be treated as separate raw datasets.

## PtRu Composition Series

### File
`PtRu_acidic_current_densities_Todd.xlsx`

### Materials
The workbook contains three worksheets corresponding to different PtRu compositions:
* `Pt100`
* `Pt66Ru33`
* `Pt50Ru50`

### HClO4 concentration
Measurements were performed at:
* 0.10 M HClO4
* 0.25 M HClO4
* 0.50 M HClO4
* 1.00 M HClO4

### CO composition
Measurements were performed at the following CO percentages:
* 0.1%
* 1%
* 10%
* 100%

The workbook labels this variable as `P_CO (%)`.

### Potential and Current Density
Each experimental condition is stored as a pair of adjacent columns:
`E_RHE (V)` | `j (mA/cm2)`, where:
* `E_RHE (V)` is electrode potential versus RHE in V.
* `j (mA/cm2)` is current density in mA/cm².

Each material contains 16 experimental conditions: `4 HClO4 concentrations × 4 CO compositions`
The potential grid and number of measurements vary between experimental conditions.
Across the workbook, potentials span approximately: 0.5 to 1 V_RHE

## Pt and Pt/ATO Dataset

### File
`Pt100_PtATO_100mM_acidic_current_densities_Em.xlsx`

### Materials and Replicates
The workbook contains six worksheets:
* `20% Pt_C Rep 1`
* `20% Pt_C Rep 2`
* `20% Pt_C Rep 3`
* `20% Pt_ATO Rep 1`
* `6% Pt_ATO Rep 1`
* `6% Pt_ATO Rep 2`

Therefore, the workbook contains:
* 3 replicates for `20% Pt_C`
* 1 replicate for `20% Pt_ATO`
* 2 replicates for `6% Pt_ATO`

### HClO4 concentration
All measurements were performed at: `100 mM HClO4`

### CO composition
Measurements were performed at:
* 0.1%
* 0.3%
* 1%
* 3%
* 10%
* 30%
* 100%

The workbook labels this variable as `PCO (%)`.

### Potential and Current Density
Each experimental condition is stored as a pair of adjacent columns:
`E_RHE (V)` | `j (mA/cm²)`, where:
* `E_RHE (V)` is electrode potential versus RHE in V.
* `j (mA/cm²)` is current density in mA/cm².

Each worksheet contains seven experimental conditions, one for each CO composition.
The potential vector contains 200 points spanning: 0.55 to 0.95 V_RHE.

## Missing Current-Density Values
In `Pt100_PtATO_100mM_acidic_current_densities_Em.xlsx`, the potential columns contain 200 values per experimental condition, but some corresponding current-density cells are blank.
The current-density values at the endpoints `0.55 V_RHE` and `0.95 V_RHE` are blank throughout the workbook. Some conditions also contain one or two additional blank current-density values near the upper end of the potential range.

## Notes
The two workbooks use potential versus RHE. Any conversion to another reference scale should likewise be performed and documented during data processing. Percentages refer to mole fractions of CO in the cell headspace. Total pressure is ambient pressure. 
