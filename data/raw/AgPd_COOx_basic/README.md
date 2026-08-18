# AgPd CO Oxidation — Basic Media

## Contents
This directory contains the raw electrochemical CO oxidation data for the AgPd catalyst series measured in KOH.
There are 6 Excel workbooks with the naming convention: `{material}_current_densities.xlsx` where `{material}` identifies the AgPd composition.

## Experimental Conditions

### KOH concentration
Measurements were performed at:
- 0.25 M KOH
- 0.50 M KOH
- 1.00 M KOH

### CO composition
Measurements were performed using:
- 0.1% CO
- 1% CO
- 10% CO
- 100% CO
Each Excel workbook contains four worksheets, one for each CO composition.
Percentages refer to mole fractions of CO in the cell headspace. Total pressure is ambient atmospheric pressure. 

### Potential
Potential is reported versus SHE.
Potential range: `-0.25 V_SHE to +0.15 V_SHE`
Each potential sweep contains 1000 points over this interval.

## Replicates
Each experimental condition contains 3 replicate measurements.
An experimental condition is defined by: `material × KOH concentration × CO composition`

## Excel File Structure
The first column contains potential versus SHE.
The remaining columns contain current-density measurements for the different KOH concentrations and experimental replicates.
Column naming convention: `{KOH_conc}M lnj uA/cm2 ({A/B/C})`
For example: `0.25M lnj uA/cm2 (B)`
The current-density columns contain `ln(j)`, where `j` is the numerical current density expressed in µA/cm² before taking the logarithm.