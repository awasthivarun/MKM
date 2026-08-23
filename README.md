# MKM

Bayesian microkinetic modeling repository for electrochemical CO oxidation.

## Project purpose

This repository implements Bayesian microkinetic modeling workflows for electrochemical CO oxidation, currently focused on AgPd in basic media. The architecture is intended to support additional catalyst systems as they are rebuilt and validated.

## Repository status

Active development is currently on the `rebuild/from-scratch` branch. Ag10Pd90 is currently the most developed posterior-analysis case.

For detailed scientific and development status, see `CURRENT_STATE.md`.

## Repository layout

- `config/`: preprocessing and model/likelihood/prior configuration.
- `data/`: raw and processed experimental tables.
- `scripts/`: reproducible workflow entry points and diagnostics.
- `src/mkm/`: core scientific and inference implementation.
- `tests/`: unit and integration tests.
- `results/`: generated inference and diagnostic outputs.
- `figures/`: generated figures.

## Basic workflow

raw data  
→ preprocessing  
→ processed tables  
→ model-data construction  
→ mechanism + priors  
→ PyMC model  
→ posterior inference  
→ derived diagnostics/post-processing

## Important architecture principle

- `src/mkm/mechanisms/` contains the mathematical/chemical mechanism implementations.
- `src/mkm/models/` contains fit-ready model definitions and registry logic that connect mechanism evaluators, parameter dataclasses, and configured prior profiles.

This separation is intentional: mechanism equations and fit-assembly metadata are related but distinct concerns.

## Development principle

Scientific calculations should live in `src/mkm/`, while `scripts/` should ideally remain thin, reproducible workflow entry points.

Some current scripts still contain mixed calculation and plotting logic and are documented as future refactor targets rather than changed during housekeeping.

## Typical commands

python scripts/process_agpd_basic.py
python scripts/plot_agpd_basic.py
python scripts/fit_agpd_posterior.py BF_LH --likelihood setup_intercept
pytest tests -v