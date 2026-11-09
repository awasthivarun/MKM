
# MKM

Bayesian microkinetic modeling of electrochemical CO oxidation.

The project investigates CO oxidation on two catalyst families:

- **AgPd in basic media:** Microkinetic modeling, Bayesian inference, and mechanistic discrimination.
- **Pt and PtRu in acidic and basic media:** Experimental preprocessing and kinetic analysis, followed by microkinetic modeling.

The project also develops generalizable methods for kinetic parameter estimation, uncertainty quantification, model comparison, and mechanistic discrimination.

## Documentation

The primary repository documentation is:

- [`CURRENT_STATE.md`](CURRENT_STATE.md): Implementation status, scientific conventions, mechanisms, and limitations.
- [`REPO_MAP.md`](REPO_MAP.md): Architecture, code ownership, data contracts, and output structure.
- [`scripts/README.md`](scripts/README.md): Command-line workflows.

Mathematical specifications and conventions take precedence over implementation descriptions. Source code establishes actual implemented behavior. Documentation may lag behind ongoing development.

## Scientific systems

### AgPd CO oxidation — basic media

Materials:

- Pd100
- Ag10Pd90
- Ag25Pd75
- Ag50Pd50
- Ag75Pd25
- Ag90Pd10

The implemented finite-rate CO models include reversible CO adsorption/desorption on Pd, Pd-OH and Ag-OH quasi-equilibria, and selectable bifunctional (BF), Eley-Rideal (ER), and Langmuir-Hinshelwood (LH) oxidation pathways.

The CO steady state is solved analytically rather than with an iterative numerical solver. Positive rates and related quantities are evaluated in the log domain where practical.

The active likelihood is Normal in linear rate space:

$$
r_i^{\mathrm{obs}} \sim
\mathcal{N}\left(r_i^{\mathrm{model}},\sigma_i^2\right),
\qquad
\sigma_i =
\sigma_{\mathrm{rate,abs}}
+\sigma_{\mathrm{rate,rel}}r_i^{\mathrm{model}}.
$$

All-material fits support `shared` and `material` error structures. Individual-material fits use one material-indexed error pair.

All-material composition fits support parameterizations configured in `config/models/agpd_basic.yaml`, including `shared` and `linear_xAg`.

Implemented analysis includes:

- Individual-material and all-material Bayesian fits.
- Prior and posterior predictive analysis.
- Posterior diagnostics and residual analysis.
- Apparent transfer coefficients and reaction orders.
- Pathway-resolved rates, coverages, and DRCs.
- Model comparison.
- Refit-based LOCO and LOMO predictive validation.

The AgPd preprocessing pipeline includes explicit data-selection and truncation rules.

### PtRu CO oxidation — acidic and basic media

Materials:

- Pt100
- Pt66Ru33
- Pt50Ru50

The primary experimental datasets are Todd's Pt/PtRu measurements in HClO4 and KOH.

Additional Pt/C measurements from Em are processed separately. Pt/ATO is not yet included in the active preprocessing workflow.

**Current status:** Experimental preprocessing, rate visualization, and kinetic-observable calculations are implemented. PtRu microkinetic mechanisms and Bayesian fitting workflows have not yet been implemented.

#### Experimental conditions

Todd's datasets contain:

- Electrolyte concentrations: 0.10, 0.25, 0.50, and 1.00 M.
- CO mole fractions: 0.001, 0.01, 0.10, and 1.00.
- Three catalyst compositions in both acidic and basic media.

Replicate coverage is uneven:

- Todd acidic: one trace per experimental condition.
- Todd basic PtRu alloys: one trace per condition.
- Todd basic Pt100: four trials per condition.

Em's Pt/C datasets contain seven CO mole fractions at 0.10 M electrolyte. Distinct catalyst preparations are retained separately.

#### Preprocessing conventions

The current PtRu preprocessing uses:

- Temperature: 293.15 K.
- Canonical potential reference: SHE.
- Ideal molarity-based H+ and OH- activities.
- Water ion product specified by `pKw: 14.167`.
- Current normalization using Pt CO-stripping ECSA.
- A charge density of 210 microC/cm2 per Pt-site electron.
- Two electrons per CO oxidation event.
- PCHIP interpolation of log rate onto a 0.01 V SHE grid.
- No extrapolation outside experimental potential ranges.

The current implementation treats CO-stripping ECSA as predominantly Pt-accessible sites for Todd's PtRu catalysts. This is an experimental normalization assumption, not proof that Ru-bound CO is absent under all operating conditions.

The assumed area basis for Em's Pt/C current densities should be verified before quantitative comparisons with Todd's data.

For subsequent kinetic modeling, the intended CO activity convention matches AgPd: ideal gas behavior with a 1 bar standard pressure and nominal total pressure of 1 bar.

#### Experimental kinetic observables

The preprocessing pipeline calculates:

**Apparent transfer coefficient**

$$
\alpha =
\frac{RT}{F}
\frac{\partial\ln r}{\partial E_{\mathrm{SHE}}}.
$$

Calculated separately for each experimental trace using second-order finite differences.

**CO reaction order**

$$
\delta_{\mathrm{CO}} =
\left.
\frac{\partial\ln r}{\partial\ln a_{\mathrm{CO}}}
\right|_{E,a_{\mathrm{ion}}}.
$$

Estimated over adjacent CO-pressure intervals using logarithms of condition-mean rates.

**Electrolyte reaction order**

$$
\delta_i =
\left.
\frac{\partial\ln r}{\partial\ln a_i}
\right|_{E,a_{\mathrm{CO}}},
\qquad
i \in \{\mathrm{H^+},\mathrm{OH^-}\}.
$$

Estimated using one log-rate versus log-concentration regression across all four electrolyte concentrations.

Unlike AgPd, PtRu CO-pressure and electrolyte-concentration trials are not assumed to be paired. Available replicate variability is propagated using an independence assumption.

Experimental observables are diagnostic quantities derived from measured rates, not additional independent likelihood observations.

#### Current limitations

- PtRu data selection and truncation have not been finalized.
- Current observable calculations use the full, untruncated analysis datasets.
- Acidic Pt100 contains sharp transitions whose interpretation remains unresolved; possible non-CO-oxidation contributions require investigation.
- Ideal electrolyte activities are used rather than literature activity-coefficient corrections.
- Replicate availability differs substantially between datasets.
- PtRu mechanisms, priors, likelihood integration, and inference workflows are not yet implemented.

## Environment

The environment is defined in `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate mkm
python -m pip install -e . --no-deps
pytest tests -q
```

`--no-deps` is intentional when the Conda environment has already installed the scientific dependencies.

## Common workflows

### AgPd preprocessing

```powershell
python scripts/process_agpd_basic.py
python scripts/plot_agpd_basic.py
```

### AgPd Bayesian inference

Fit an individual material:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER --material Ag50Pd50
```

Run posterior diagnostics and DRC:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER --material Ag50Pd50 --plot-level full
python scripts/postprocess_agpd_drc.py CO_BF_ER --material Ag50Pd50 --check-half-step
```

Fit an all-material composition model:

```powershell
python scripts/fit_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material
```

Postprocess and analyze the composition fit:

```powershell
python scripts/postprocess_agpd_posterior.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --plot-level full
python scripts/postprocess_agpd_drc.py CO_BF_ER_LH --all-materials --parameterization linear_xAg --error-structure material --check-half-step
python scripts/plot_agpd_composition_parameters.py CO_BF_ER_LH --parameterization linear_xAg --error-structure material
```

### PtRu preprocessing

Standardize and interpolate Todd's Pt/PtRu data:

```powershell
python scripts/process_ptru_todd.py
```

Standardize and interpolate Em's Pt/C data:

```powershell
python scripts/process_ptru_em.py
```

Generate experimental rate figures:

```powershell
python scripts/plot_ptru_todd.py
```

Calculate and plot all experimental kinetic observables:

```powershell
python scripts/process_ptru_observables.py
```

The consolidated observable entry point replaces the earlier separate alpha, CO-order, and electrolyte-order scripts. The old scripts may be retained temporarily for regression verification.

PtRu figures are exported as both PNG and SVG.

## Repository structure

```text
config/
    preprocessing/        Experimental data-processing settings
    models/               Microkinetic model configuration
    priors/               Bayesian prior specifications

data/
    raw/                  Original experimental data
    processed/            Standardized and analysis datasets

figures/
    preprocessing/        Experimental preprocessing figures

results/                  Inference, validation, and postprocessing outputs

scripts/                  Command-line workflow entry points

src/mkm/
    preprocessing/        Data processing, observables, and plotting
    mechanisms/           Elementary-step kinetic formulations
    models/               Mechanism assembly and parameterization
    inference/            Priors, likelihoods, and posterior sampling
    postprocessing/       Diagnostics, observables, DRCs, and comparisons
    workflows/            Higher-level workflow orchestration

tests/                    Unit, integration, and workflow tests

CURRENT_STATE.md          Scientific and implementation status
REPO_MAP.md               Architecture and code ownership
```

### PtRu preprocessing modules

The PtRu pipeline uses:

- `ptru_todd.py`: Todd workbook ingestion, standardization, and interpolation.
- `ptru_em.py`: Em Pt/C workbook ingestion and standardization.
- `ptru_observables.py`: PtRu-specific observable calculations and grouping conventions.
- `ptru_plotting.py`: Rate and observable figures using shared AgPd styling utilities.

The scientific calculations and plotting implementations reside in `src/mkm/preprocessing/`. The scripts provide command-line orchestration.

## Interpretation boundary

Statistical fit does not by itself establish mechanistic validity.

Converged sampling, narrow posteriors, favorable predictive scores, and nonzero pathway contributions must be evaluated alongside physical consistency, parameter identifiability, residual structure, surface coverages, pathway rates, and degrees of rate control.

Derived kinetic observables can reveal mechanistically informative behavior, but multiple peaks, negative reaction orders, or unusual apparent transfer coefficients do not uniquely identify elementary mechanisms.

Where experiments cannot distinguish competing mechanistic explanations, that uncertainty must be reported explicitly.
