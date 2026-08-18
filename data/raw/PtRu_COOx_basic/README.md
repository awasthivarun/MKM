# PtRu CO Oxidation — Basic Media

## Contents
This directory contains raw electrochemical CO oxidation current-density data measured on Pt and PtRu catalysts in KOH.
Three Excel workbooks are included:
* `Pt100_basic_current_densities_Todd.xlsx`
* `PtRu_basic_current_densities_Todd.xlsx`
* `Pt100_100mM_basic_current_densities_Em.xlsx`

The three workbooks contain different catalyst sets, replicate structures, and experimental condition grids and should be treated as separate raw datasets.

## Pt100 KOH Series

### File
`Pt100_basic_current_densities_Todd.xlsx`

### Material
The workbook contains current-density measurements for: `Pt100`

### KOH concentration
Measurements were performed at:
* 0.10 M KOH
* 0.25 M KOH
* 0.50 M KOH
* 1.00 M KOH

### CO composition
Measurements were performed at:
* 0.1% CO
* 1% CO
* 10% CO
* 100% CO

### Replicates
Four trials are present for each KOH concentration:
* Trial 1
* Trial 2
* Trial 3
* Trial 4

Each trial is stored in a separate worksheet. There are therefore 16 current-density worksheets: `4 KOH concentrations × 4 trials`
Each worksheet contains all four CO compositions. The potential grids and number of measurements vary between trials and experimental conditions.

### Excel File Structure
Within each current-density worksheet, each CO condition is stored as a pair of adjacent columns:
`E (V_RHE)` | `j (mA/cm2)` where:
* `E (V_RHE)` is electrode potential versus RHE in V.
* `j (mA/cm2)` is current density in mA/cm².

Across the current-density worksheets, the measured potentials span approximately: 0.5 to 1 V_RHE

### Additional Worksheets
The workbook also contains the following derived-data worksheets:
* `TC`
* `CO order`
* `OH- Order`

These worksheets are not considered part of the raw current-density dataset for the present project and should be ignored during current-density preprocessing. Transfer coefficients and reaction orders will instead be calculated from the current-density data during processing.

## PtRu KOH Series

### File
`PtRu_basic_current_densities_Todd.xlsx`

### Materials
The workbook contains two worksheets corresponding to different PtRu compositions:
* `Pt66Ru33`
* `Pt50Ru50`

### KOH concentration
Measurements were performed at:
* 0.10 M KOH
* 0.25 M KOH
* 0.50 M KOH
* 1.00 M KOH

### CO composition
Measurements were performed at:
* 0.1% CO
* 1% CO
* 10% CO
* 100% CO

The workbook labels this variable as `P_CO (%)`.

### Excel File Structure
Each experimental condition is stored as a pair of adjacent columns:
`E_RHE (V)` | `j (mA/cm2)` where:
* `E_RHE (V)` is electrode potential versus RHE in V.
* `j (mA/cm2)` is current density in mA/cm².

Each material contains 16 experimental conditions: `4 KOH concentrations × 4 CO compositions`
The potential grid and number of measurements vary between experimental conditions.
Across the workbook, potentials span approximately: 0.5 to 1 V_RHE
No replicate dimension is explicitly represented in this workbook; each KOH concentration and CO composition combination contains one potential/current-density trace.

## Pt100 100 mM KOH Dataset

### File
`Pt100_100mM_basic_current_densities_Em.xlsx`

### Material and Worksheets
The workbook contains three worksheets:
* `20% Pt_C Rep 1`
* `20% Pt_C 1mg Rep 1`
* `20% Pt_C 1mg Rep 3`

The worksheet labels are retained as provided in the raw workbook.

### KOH concentration
All measurements were performed at: `100 mM KOH`

### CO composition
Measurements were performed at:
* 0.1% CO
* 0.3% CO
* 1% CO
* 3% CO
* 10% CO
* 30% CO
* 100% CO

The workbook labels this variable as `PCO (%)`.

### Potential and Current Density
Each experimental condition is stored as a pair of adjacent columns:
`E_RHE (V)` | `j (mA/cm²)` where:
* `E_RHE (V)` is electrode potential versus RHE in V.
* `j (mA/cm²)` is current density in mA/cm².

Each worksheet contains seven experimental conditions, one for each CO composition. Each potential vector contains 200 points spanning: 0.55 to 0.95 V_RHE

### Missing Current-Density Values
The potential vectors contain values for the complete 200-point grid, but some corresponding current-density cells are blank.
For `20% Pt_C Rep 1`, current density is blank at the two endpoints:
* `0.55 V_RHE`
* `0.95 V_RHE`

For both `20% Pt_C 1mg Rep 1` and `20% Pt_C 1mg Rep 3`, current density is blank at:
* `0.55 V_RHE`
* potentials from approximately `0.928 V_RHE` through `0.95 V_RHE`

These missing values occur consistently across the seven CO compositions within the corresponding worksheets. The missing cells should remain unchanged in the raw workbook and should be handled explicitly during data processing.

## Notes
The two workbooks use potential versus RHE. Any conversion to another reference scale should likewise be performed and documented during data processing. Percentages refer to mole fractions of CO in the cell headspace. Total pressure is ambient pressure. 
