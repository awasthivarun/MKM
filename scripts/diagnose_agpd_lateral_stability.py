"""Refine and classify Frumkin steady states using the finite-rate kinetic ODEs.

Read-only: reuses saved prior parameters and experimental conditions. Requires all CO,
pOH, sOH, and step 3 to have finite kinetics. Does not alter the production solver.
"""

from argparse import ArgumentParser
from dataclasses import fields
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, softmax
import xarray as xr

from mkm.constants import H_J_S, K_B_EV_K, K_B_J_K
from mkm.mechanisms.agpd_basic.state import build_agpd_point_state
from mkm.model_inputs import build_model_input_arrays, build_model_point_inputs
from mkm.models.agpd_physics import parse_agpd_physics
from mkm.models.agpd_registry import get_agpd_model_definition
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config


def kinetic_balances(theta, parameters, condition, temperature_K, bep_coefficient):
    """Original finite-rate ODE right-hand side for (CO, pOH, COOH, sOH) coverages.

    These equations mirror lateral_interactions.py. The primary vacancy is 1-CO-pOH-COOH.
    Returns ODE rates and positive turnover denominators for dimensionless residuals.
    """
    c, p, h, s = theta
    e = 1.0 - c - p - h
    E, ln_a_co, ln_a_oh, fp, fs = condition
    q = parameters["q"]
    zc, zx, zo = (parameters[name] for name in ("z_CO_CO_eV", "z_CO_OH_eV", "z_OH_OH_eV"))
    kbt = K_B_EV_K * temperature_K
    log_prefactor = np.log(K_B_J_K * temperature_K / H_J_S)
    C, H = fp * c, fp * p + q * fs * s
    mu_c, mu_p = zc * C + zx * H, zx * C + zo * H
    mu_s = q * mu_p
    dg1 = parameters["deltaG1_0"] + mu_c
    dg4 = parameters["deltaG4_0"] - E + mu_p
    dg5 = parameters["deltaG5_0"] - q * E + mu_s
    dg3 = parameters["deltaG3_0"] - E

    def k(g, ln_activity=0.0):
        return np.exp(log_prefactor - g / kbt + ln_activity)

    gamma = bep_coefficient
    k1a = k(parameters["Gact1_0"] + gamma * mu_c, ln_a_co)
    k1b = k1a * np.exp(dg1 / kbt - ln_a_co)
    k4a = k(parameters["Gact4_0"] - parameters["beta_4"] * E + gamma * mu_p, ln_a_oh)
    k4b = k4a * np.exp(dg4 / kbt - ln_a_oh)
    k5a = k(parameters["Gact5_0"] - parameters["beta_5"] * q * E + gamma * mu_s, ln_a_oh)
    k5b = k5a * np.exp(dg5 / kbt - ln_a_oh)
    kbf = k(parameters["Gact2_BF_0"] - parameters["beta_2_BF"] * (1.0 - q) * E
            - gamma * (mu_c + mu_s))
    kbfr = kbf * np.exp((dg3 - dg1 - dg5) / kbt)
    ker = k(parameters["Gact2_ER_0"] - parameters["beta_2_ER"] * E - gamma * mu_c, ln_a_oh)
    kerr = ker * np.exp((dg3 - dg1) / kbt - ln_a_oh)
    klh = k(parameters["Gact2_LH_0"] - gamma * (mu_c + mu_p))
    klhr = klh * np.exp((dg3 - dg1 - dg4) / kbt)
    k3a = k(parameters["Gact3_0"] - parameters["beta_3"] * E, ln_a_oh)

    rbf = fs * (kbf * c * s - kbfr * h * (1.0 - s))
    rer = ker * c - kerr * h
    rlh = fp * (klh * c * p - klhr * h * e)
    total = rbf + rer + rlh
    transfer_s = fp * rbf / fs
    rates = np.asarray([
        k1a * e - k1b * c - total,
        k4a * e - k4b * p - rlh,
        total - k3a * h,
        k5a * (1.0 - s) - k5b * s - transfer_s,
    ])
    denominators = np.asarray([
        k1a * e + k1b * c + abs(total),
        k4a * e + k4b * p + abs(rlh),
        abs(total) + k3a * h,
        k5a * (1.0 - s) + k5b * s + abs(transfer_s),
    ])
    return rates, denominators


def to_unconstrained(theta):
    """Four unconstrained log-ratios representing a physical two-site coverage state."""
    c, p, h, s = np.asarray(theta, dtype=float)
    e = max(1.0 - c - p - h, 1e-15)
    return np.log(np.maximum([c, p, h], 1e-15) / e).tolist() + [np.log(np.clip(s, 1e-15, 1 - 1e-15) /
                                                                           np.clip(1 - s, 1e-15, 1))]


def from_unconstrained(y):
    c, p, h, _ = softmax(np.array([y[0], y[1], y[2], 0.0]))
    return np.asarray([c, p, h, expit(y[3])])


def balance_error(theta, parameters, condition, temperature_K, bep_coefficient):
    rates, denominators = kinetic_balances(theta, parameters, condition, temperature_K, bep_coefficient)
    return rates / np.maximum(denominators, 1e-300)


def physical_jacobian(theta, parameters, condition, temperature_K, bep_coefficient):
    """Complex-step derivative of the physical ODE, not the numerical fixed-point iteration."""
    step = 1e-24
    jacobian = np.empty((4, 4))
    for i in range(4):
        displaced = np.asarray(theta, dtype=np.complex128)
        displaced[i] += 1j * step
        rates, _ = kinetic_balances(displaced, parameters, condition, temperature_K, bep_coefficient)
        jacobian[:, i] = np.imag(rates) / step
    return jacobian


def analyze_stationary_state(theta, parameters, condition, temperature_K, bep_coefficient, tolerance=1e-7):
    """Refine one starting point and classify stability only if the balances meet tolerance."""
    initial = to_unconstrained(theta)

    def residual(y):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = balance_error(from_unconstrained(y), parameters, condition, temperature_K, bep_coefficient)
        return np.nan_to_num(values, nan=1e12, posinf=1e12, neginf=-1e12)

    optimized = least_squares(residual, initial, max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    theta_refined = from_unconstrained(optimized.x)
    error = float(np.max(np.abs(residual(optimized.x))))
    result = {"theta": theta_refined, "error": error, "nfev": optimized.nfev, "root": error <= tolerance}
    if result["root"]:
        eigenvalues = np.linalg.eigvals(physical_jacobian(theta_refined, parameters, condition,
                                                           temperature_K, bep_coefficient))
        result["eigenvalues"] = eigenvalues
        result["stability"] = ("unstable" if np.max(eigenvalues.real) > 1e-8 else
                               "stable" if np.max(eigenvalues.real) < -1e-8 else "marginal")
    else:
        result["eigenvalues"] = None
        result["stability"] = "unresolved"
    return result



def find_distinct_roots(parameters, condition, temperature_K, bep_coefficient, n_starts=41):
    """Search several physically admissible initial states; not a proof of all roots."""
    initial_states = [(0.88, 0.05, 0.01, 0.10), (0.05, 0.88, 0.01, 0.90)]
    initial_states += [(c, (1.0-c)*0.9999, (1.0-c)*1e-5, 0.01)
                       for c in np.linspace(0.02, 0.98, n_starts)]
    roots = []
    for theta in initial_states:
        candidate = analyze_stationary_state(theta, parameters, condition, temperature_K, bep_coefficient)
        if candidate["root"] and not any(np.max(abs(candidate["theta"]-old["theta"])) < 1e-5 for old in roots):
            roots.append(candidate)
    return sorted(roots, key=lambda item: item["theta"][0])


def load_saved_parameter_draw(prior, model_name, chain, draw):
    """Read exact physical parameters from the saved prior, without triggering inference or scanning."""
    definition = get_agpd_model_definition(model_name)
    values = {}
    for field in fields(definition.parameter_class):
        if field.name not in prior:
            raise ValueError(f"Saved prior missing physical parameter '{field.name}'.")
        array = np.asarray(prior[field.name].values[chain, draw], dtype=float)
        if array.ndim != 0:
            raise ValueError(f"Expected scalar physical parameter '{field.name}', got shape {array.shape}.")
        values[field.name] = float(array)
    return values


def cooh_removal_rate(theta, parameters, condition, temperature_K):
    """Net CO oxidation turnover through finite, irreversible step 3 [1/s]."""
    E, _, ln_a_oh, _, _ = condition
    barrier = parameters["Gact3_0"] - parameters["beta_3"] * E
    return (K_B_J_K * temperature_K / H_J_S) * np.exp(-barrier / (K_B_EV_K * temperature_K) + ln_a_oh) * theta[2]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="cxo_CO_BF_ER_LH_3_pOH_sOH")
    parser.add_argument("--material", default="Ag50Pd50")
    parser.add_argument("--point-ids", nargs="+", type=int, default=[115])
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--draw", type=int, default=2)
    parser.add_argument("--starts", type=int, default=41, help="Number of interior initial coverages to search")
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("--starts must be positive.")
    physics = parse_agpd_physics(args.model)
    if set(physics.finite_steps) != {"CO", "pOH", "sOH", "3"} or set(physics.pathways) != {"BF", "ER", "LH"}:
        parser.error("This stability diagnostic requires finite CO, pOH, sOH, 3 and active BF, ER, LH.")
    if set(physics.interactions) != {"c", "x", "o"}:
        parser.error("This diagnostic is limited to c/x/o interaction models.")

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    output_dir = paths.agpd_prior_predictive_output_dir(fit_scope="individual", model_name=args.model,
                                                         material=args.material)
    saved = output_dir / "prior_predictive.nc"
    model_data = build_agpd_model_data(paths, (args.material,))
    point_inputs = build_model_point_inputs(build_model_input_arrays(model_data))
    state = build_agpd_point_state(point_inputs, config)
    ids = np.asarray(args.point_ids, dtype=int)
    if np.any(ids < 0) or np.any(ids >= len(model_data.model_points)):
        parser.error("At least one point ID is outside the model-point table.")
    gamma = config["lateral_interactions"]["bep_coefficient"]
    temp = config["temperature_K"]
    print(f"Saved parameter draw: {saved}, chain={args.chain}, draw={args.draw}")
    with xr.open_dataset(saved, group="prior", engine="h5netcdf") as prior:
        parameters = load_saved_parameter_draw(prior, args.model, args.chain, args.draw)
    print("Independent scipy root search (not a change to the production solver).")
    for point_id in ids:
        condition = (float(state.E_V_SHE[point_id]), float(state.ln_a_CO[point_id]),
                     float(state.ln_a_OH[point_id]), float(state.Pd_fraction[point_id]),
                     float(state.Ag_fraction[point_id]))
        print(f"\nPoint {point_id}: E={condition[0]:.6g} V, aCO={np.exp(condition[1]):.6g}, "
              f"aOH={np.exp(condition[2]):.6g}, fp={condition[3]:.6g}, fs={condition[4]:.6g}")
        if condition[3] <= 0 or condition[4] <= 0:
            print("Skipped: this four-variable dynamic system requires positive populations of both site types.")
            continue
        roots = find_distinct_roots(parameters, condition, temp, gamma, args.starts)
        print(f"Distinct physical roots found: {len(roots)}")
        for j, result in enumerate(roots):
            c, p, h, s = result["theta"]
            largest = np.max(result["eigenvalues"].real)
            rate = cooh_removal_rate(result["theta"], parameters, condition, temp)
            print(f"Root {j}: CO={c:.8f}, pOH={p:.8f}, COOH={h:.6g}, sOH={s:.8f}, "
                  f"vacant={1-c-p-h:.6g}, residual={result['error']:.3g}, "
                  f"max_Re_eigenvalue={largest:.6g} /s, {result['stability']}, rate={rate:.6g} /s")
            print("  Eigenvalues [1/s]: " + ", ".join(f"{value:.6g}" for value in result["eigenvalues"]))
        if sum(item["stability"] == "stable" for item in roots) > 1:
            print("MULTIPLE LOCALLY STABLE STATES: physical branch-selection decision is required.")
        print("Local stability is based on the finite-rate ODE Jacobian; root search is not exhaustive.")


if __name__ == "__main__":
    main()
