# CO Oxidation Model Registry

## Purpose

This file is the authoritative catalog of mechanistic models used in the CO oxidation project.

The registry separates:

- mechanism identity,
- mathematical formulation,
- likelihood/error model,
- material/environment applicability,
- fitting status,
- scientific interpretation.

A model should not be considered a distinct mechanism merely because it was run with a different prior or error model.

---

## Model status definitions

### Screening
Model is being tested primarily to determine whether it can plausibly describe the experimental behavior.

### Finalist
Model has survived initial screening and is being evaluated with the preferred statistical treatment and full diagnostics.

### Rejected
Model has been ruled out for a documented physical, mathematical, statistical, or numerical reason.

### Mega
Model is formulated to simultaneously describe multiple materials.

---

## Evidence categories

Conclusions from a model should be labeled using:

1. Mathematically established
2. Supported mechanistic interpretation
3. Plausible but unresolved hypothesis
4. Insufficient information to conclude

Good statistical fit alone is not evidence that a mechanism is correct.

---

# Model Registry

| ID | Name | Environment | Pathway type | Stage | Notes |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | Populate during notebook refactor |

---

# Model Entry Template

Each distinct mechanistic model should receive one entry using the format below.

## MODEL-ID — Model Name

### Scientific scope

- Catalyst family:
- Environment:
- Materials:
- Current status:

### Elementary steps

1. 
2. 
3. 

Identify each step as appropriate:

- QEA
- RDS
- fast
- irreversible
- reversible

### Rate expression

Write the final rate expression mathematically.

### Site balances

Document all site types and balances.

Example:

`theta_empty + theta_CO + theta_OH = 1`

### Free parameters

| Parameter | Meaning | Units | Domain |
|---|---|---|---|
|  |  |  |  |

### Priors

Priors belong to a particular fitting configuration and may change without creating a new mechanistic model.

| Parameter | Distribution | Hyperparameters | Rationale |
|---|---|---|---|
|  |  |  |  |

### Potential dependence

Document all electrochemical free-energy and activation-energy relationships explicitly.

### Concentration and pressure dependence

Document the standard-state convention and all factors involving:

- CO pressure
- proton concentration
- hydroxide concentration

### Composition dependence

For mega-models, document all dependence on catalyst composition.

Distinguish explicitly between:

1. site-abundance/topological effects;
2. electronic changes in adsorption energies;
3. electronic changes in activation barriers;
4. empirical composition relationships.

### Likelihood / error model

Document separately from the mechanism:

- observed quantity
- distribution
- experimental uncertainty contribution
- additional model discrepancy term

### Deterministic observables

Examples:

- log_rate
- coverages
- alpha
- delta_CO
- delta_OH
- delta_H

### DRC variables

List the physical elementary-step quantities that should be perturbed for degree-of-rate-control calculations.

### Physical constraints

Document constraints such as:

- coverage bounds
- site balance
- positive activation barriers
- bounded transfer coefficients
- thermodynamic consistency

### Known identifiability issues

Document strong posterior correlations, weakly identified quantities, or parameter combinations that produce equivalent model behavior.

### Results by material

| Material | Screening | Finalist run | Result / reason |
|---|---|---|---|
|  |  |  |  |

### Scientific interpretation

Separate:

- what is mathematically required;
- what the data statistically support;
- what may be mechanistically inferred;
- what remains unresolved.

### Rejection reason

If rejected, classify the primary reason:

- physical-model failure
- statistical-model failure
- numerical-inference failure
- identifiability failure
- inadequate predictive behavior
- other