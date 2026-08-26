from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


DATA_PATH = Path(
    "data/processed/AgPd_COOx_basic/analysis/"
    "AgPd_COOx_basic_selected.parquet"
)

RATE_COL = "rate_s_inv"

GROUP_COLS = [
    "material",
    "C_KOH_M",
    "CO_mole_fraction",
    "E_V_SHE",
]


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

data = pd.read_parquet(DATA_PATH)

print("Columns:")
print(data.columns.tolist())

missing = [c for c in [*GROUP_COLS, RATE_COL] if c not in data.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")


# ---------------------------------------------------------------------
# Collapse experimental replicates
# ---------------------------------------------------------------------

rep = (
    data.groupby(GROUP_COLS, as_index=False)[RATE_COL]
    .agg(
        n_replicates="count",
        mean_rate="mean",
        sample_sd="std",
        sample_var="var",
    )
)

rep = rep.loc[
    (rep["n_replicates"] >= 2)
    & np.isfinite(rep["mean_rate"])
    & np.isfinite(rep["sample_var"])
    & (rep["sample_var"] > 0)
].copy()

print("\nReplicate counts:")
print(rep["n_replicates"].value_counts().sort_index())

print(f"\nNumber of replicate groups: {len(rep)}")


# ---------------------------------------------------------------------
# Likelihood for observed replicate variances
#
# If replicate measurements are Normal:
#
#   (n - 1) * s^2 / sigma^2 ~ chi-square(n - 1)
#
# This is preferable to least-squares fitting the noisy sample SDs.
# ---------------------------------------------------------------------

rate = rep["mean_rate"].to_numpy(float)
s2 = rep["sample_var"].to_numpy(float)
dof = rep["n_replicates"].to_numpy(float) - 1.0


def variance_loglik(sigma2):
    sigma2 = np.asarray(sigma2, dtype=float)

    if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return -np.inf

    z = dof * s2 / sigma2

    return np.sum(
        chi2.logpdf(z, df=dof)
        + np.log(dof / sigma2)
    )


def fit_error_model(name, variance_function, x0, bounds=None):
    def objective(params):
        return -variance_loglik(
            variance_function(params, rate)
        )

    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )

    loglik = -result.fun
    k = len(result.x)
    n = len(rep)

    return {
        "name": name,
        "params": result.x,
        "loglik": loglik,
        "AIC": 2 * k - 2 * loglik,
        "BIC": k * np.log(n) - 2 * loglik,
        "variance_function": variance_function,
        "success": result.success,
        "message": result.message,
    }


# ---------------------------------------------------------------------
# Candidate error laws
#
# Positive parameters are represented in log space.
# ---------------------------------------------------------------------

# sigma = eta
def constant_variance(params, r):
    eta = np.exp(params[0])
    return np.full_like(r, eta**2)


# sigma = omega * r
def relative_variance(params, r):
    omega = np.exp(params[0])
    return (omega * r) ** 2


# sigma = eta + omega*r
def additive_sd_variance(params, r):
    eta = np.exp(params[0])
    omega = np.exp(params[1])
    return (eta + omega * r) ** 2


# Hsu with gamma = 2:
#
# sigma^2 = eta^2 + omega^2*r^2
def quadrature_variance(params, r):
    eta = np.exp(params[0])
    omega = np.exp(params[1])
    return eta**2 + (omega * r) ** 2


# Full Hsu:
#
# sigma^2 = eta^2 + omega^2*r^gamma
def hsu_variance(params, r):
    eta = np.exp(params[0])
    omega = np.exp(params[1])
    gamma = params[2]

    return eta**2 + omega**2 * np.power(r, gamma)


# ---------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------

typical_sd = np.median(rep["sample_sd"])
typical_rate = np.median(np.abs(rate))

eta0 = max(0.5 * typical_sd, 1e-12)
omega0 = max(typical_sd / typical_rate, 1e-12)


fits = [
    fit_error_model(
        "constant",
        constant_variance,
        [np.log(typical_sd)],
    ),

    fit_error_model(
        "relative_only",
        relative_variance,
        [np.log(omega0)],
    ),

    fit_error_model(
        "additive_SD",
        additive_sd_variance,
        [np.log(eta0), np.log(omega0)],
    ),

    fit_error_model(
        "quadrature_Hsu_gamma2",
        quadrature_variance,
        [np.log(eta0), np.log(omega0)],
    ),

    fit_error_model(
        "Hsu_3_parameter",
        hsu_variance,
        [np.log(eta0), np.log(omega0), 2.0],
        bounds=[
            (None, None),
            (None, None),
            (0.0, 4.0),
        ],
    ),
]


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

rows = []

for fit in fits:
    p = fit["params"]

    if fit["name"] == "constant":
        eta = np.exp(p[0])
        omega = np.nan
        gamma = 0.0

    elif fit["name"] == "relative_only":
        eta = 0.0
        omega = np.exp(p[0])
        gamma = 2.0

    elif fit["name"] in {"additive_SD", "quadrature_Hsu_gamma2"}:
        eta = np.exp(p[0])
        omega = np.exp(p[1])
        gamma = 2.0

    else:
        eta = np.exp(p[0])
        omega = np.exp(p[1])
        gamma = p[2]

    rows.append(
        {
            "model": fit["name"],
            "sigma_abs": eta,
            "sigma_rel": omega,
            "gamma": gamma,
            "loglik": fit["loglik"],
            "AIC": fit["AIC"],
            "BIC": fit["BIC"],
            "success": fit["success"],
        }
    )

comparison = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

comparison["delta_AIC"] = comparison["AIC"] - comparison["AIC"].min()
comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()

print("\nERROR MODEL COMPARISON")
print(comparison.to_string(index=False))


# ---------------------------------------------------------------------
# Plot replicate SD against mean rate
# ---------------------------------------------------------------------

positive = (rep["mean_rate"] > 0) & (rep["sample_sd"] > 0)

r_grid = np.geomspace(
    rep.loc[positive, "mean_rate"].min(),
    rep.loc[positive, "mean_rate"].max(),
    500,
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    rep.loc[positive, "mean_rate"],
    rep.loc[positive, "sample_sd"],
    alpha=0.3,
    label="experimental replicate SD",
)

for fit in fits:
    sigma = np.sqrt(
        fit["variance_function"](
            fit["params"],
            r_grid,
        )
    )

    ax.plot(
        r_grid,
        sigma,
        label=fit["name"],
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Mean experimental rate / s$^{-1}$")
ax.set_ylabel("Replicate SD / s$^{-1}$")

ax.legend()
fig.tight_layout()

figure_dir = Path("figures")
figure_dir.mkdir(parents=True, exist_ok=True)

output_path = figure_dir / "error_model_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure to: {output_path}")