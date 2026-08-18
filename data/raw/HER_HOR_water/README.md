# Pt HER/HOR — Aqueous H2SO4

## Contents
This directory contains raw electrochemical hydrogen evolution reaction (HER) and hydrogen oxidation reaction (HOR) data measured in aqueous media.
The raw data for Pt/C working electrode with H2SO4 as proton source are contained in: `Pt_H2SO4_currents.xlsx`. 
The workbook contains one worksheet: `Sheet1`

## Electrochemical System
The geometric electrode area was reported to be constant between experiments because the same procedure was used to prepare the electrode flags.
No ECSA measurement is included in this dataset. The workbook therefore contains absolute current rather than ECSA-normalized current density.

## Experimental Variables
Each experimental condition is characterized by:
- H+ activity, `aH+`
- H2 fugacity, `fH2`
- electrode potential versus SHE
- measured current

### H+ activity
The workbook stores H+ activity under the label: `aH+`.
The precise values vary slightly between experimental sequences. The extended Debye-Hückel equation was used to calculate activities. 

### H2 fugacity
The workbook stores hydrogen fugacity under the label: `fH2`
H2 fugacity is reported in bar. Fugacity values were calculated using the Soave-Redlich-Kwong (SRK) equation of state.
The calculated fugacities differ somewhat between the two experimental sequences and are stored explicitly for each experimental condition in the workbook.

## Excel File Structure
Each experimental condition is stored as a pair of adjacent columns: `E (VSHE)` | `I (A)`, where:
- `E (VSHE)` is electrode potential versus SHE in V.
- `I (A)` is measured current in A.

The corresponding `aH+` and `fH2` values are stored above each potential/current pair. A blank column generally separates adjacent condition blocks.

## Experimental Sequences
The workbook contains two measurement sequences, referred to here as sequence A and sequence B. These should be treated as repeated experimental sequences rather than pointwise replicates because their potential grids and calculated experimental conditions are not identical.

### Sequence A
Sequence A occupies the first major block of the worksheet and contains 18 condition blocks. Each condition contains 240 potential/current measurements. The sequence nominally covers three H+ activity levels and six H2 fugacities per H+ activity.

### Sequence B
Sequence B begins at column `BE` following the blank separator between the two major data blocks. It contains 16 condition blocks. The number of potential/current measurements varies between conditions. For the two lower H+ activity levels, six H2 fugacity conditions are present. At the highest H+ activity, only four H2 fugacity conditions are present; the two lowest-fugacity conditions are absent. The H+ activities and H2 fugacities stored for sequence B differ slightly from those stored for sequence A.

## Known Data Issue
The sequence A condition in columns `S:T` is labeled with: `aH+ = 0.00522189114755799`. 
This value appears inconsistent with the organization of the surrounding conditions and should be verified against the original experimental records.

## Unresolved Metadata
The following experimental metadata should be recovered or confirmed if possible:
- Geometric electrode area
