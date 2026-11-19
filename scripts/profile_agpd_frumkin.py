"""Read-only AgPd Frumkin profiler. Outputs metrics to stdout for logging with Tee-Object.

Stages:
  stats: posterior nutpie sample_stats grouped by chain.
  coverage: compile/evaluate the complete Frumkin coverage graph at saved parameter draws.
  logp: compile/evaluate PyMC log probability and gradient at its initial point.

No inference, changes to priors, or writes to result directories occur here.
"""

import argparse
from dataclasses import fields, replace
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import xarray as xr


def elapsed(label, fn):
    start = perf_counter()
    result = fn()
    print(f"{label}: {perf_counter() - start:.3f} s", flush=True)
    return result


def print_stats(path):
    if not path.is_file():
        raise FileNotFoundError(f"Posterior sample not found: {path}")
    with xr.open_dataset(path, group="sample_stats", engine="h5netcdf") as stats:
        print(f"Sample statistics file: {path}")
        print("Variables: " + ", ".join(sorted(stats.data_vars)))
        n_chains = stats.sizes.get("chain", 0)
        n_draws = stats.sizes.get("draw", 0)
        print(f"Chains={n_chains}, retained draws per chain={n_draws}")
        for chain in range(n_chains):
            print(f"\nChain {chain} (quantiles of retained draws):")
            for name in sorted(stats.data_vars):
                array = stats[name]
                if not {"chain", "draw"}.issubset(array.dims) or array.ndim != 2:
                    continue
                if not np.issubdtype(array.dtype, np.number) and array.dtype != bool:
                    continue
                values = np.asarray(array.isel(chain=chain), dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    print(f"  {name:28s} nonfinite={values.size}")
                    continue
                p = np.percentile(finite, [5, 50, 90, 99, 100])
                print(f"  {name:28s} p05={p[0]:.4g} p50={p[1]:.4g} p90={p[2]:.4g} "
                      f"p99={p[3]:.4g} max={p[4]:.4g} nonfinite={values.size-finite.size}")
        for name in ("n_steps", "n_grad", "n_gradient", "n_leapfrog", "tree_depth", "depth"):
            if name in stats and {"chain", "draw"}.issubset(stats[name].dims):
                print(f"Across-chain {name}: " + ", ".join(
                    f"chain {i}: {float(stats[name].isel(chain=i).mean()):.4g}"
                    for i in range(n_chains)))


def saved_parameters(path, model_name, chain, draw, source):
    from mkm.models.agpd_registry import get_agpd_model_definition

    group = "posterior" if source == "posterior" else "prior"
    definition = get_agpd_model_definition(model_name)
    with xr.open_dataset(path, group=group, engine="h5netcdf") as dataset:
        if chain < 0 or chain >= dataset.sizes["chain"] or draw < 0 or draw >= dataset.sizes["draw"]:
            raise IndexError(f"Requested ({chain}, {draw}), available: "
                             f"{dataset.sizes['chain']} chains, {dataset.sizes['draw']} draws")
        params = {}
        for field in fields(definition.parameter_class):
            if field.name not in dataset:
                raise KeyError(f"Missing saved physical parameter '{field.name}' in {path}:{group}")
            value = np.asarray(dataset[field.name].isel(chain=chain, draw=draw), dtype=float)
            if value.ndim != 0:
                raise ValueError(f"Expected scalar {field.name}; got shape {value.shape}")
            params[field.name] = float(value)
    return params


def _select_point_ids(count, total):
    """Spread measurements across experimental conditions, including a known stiff point."""
    count = min(count, total)
    ids = np.linspace(0, total - 1, count, dtype=int)
    if count > 0 and total > 115 and 115 not in ids:
        ids[count // 2] = 115
    return np.sort(np.unique(ids))


def _summarize_coverages(outputs, stage):
    names = ("ln_rate", "iterations", "residual", "branch_diff", "theta_CO")
    data = dict(zip(names, (np.asarray(a, dtype=float) for a in outputs)))
    valid = np.isfinite(data["ln_rate"])
    it = data["iterations"]
    finite_iters = it[np.isfinite(it)]
    print(f"  {stage}: finite_rates={int(valid.sum())}/{valid.size}; "
          f"max_residual={float(np.nanmax(data['residual'])):.5g}; "
          f"max_branch_diff={float(np.nanmax(data['branch_diff'])):.5g}")
    if finite_iters.size:
        q = np.percentile(finite_iters, [0, 25, 50, 75, 90, 95, 99, 100])
        print("  Iterations [min,p25,p50,p75,p90,p95,p99,max]: " + ", ".join(f"{v:.1f}" for v in q))
        for cutoff in (32, 64, 96, 128, 192, 256, 384):
            print(f"  iterations > {cutoff}: {int(np.sum(finite_iters > cutoff))}/{finite_iters.size}")
    bad_ids = np.flatnonzero(~valid)
    if bad_ids.size:
        print("  Nonfinite local indices (first 15): " + str(bad_ids[:15].tolist()))
    return data


def profile_coverage(paths, config, args):
    import pytensor
    import pytensor.tensor as pt
    from mkm.mechanisms.agpd_basic.lateral_interactions import evaluate_agpd_lateral_interactions
    from mkm.mechanisms.agpd_basic.state import build_agpd_point_state
    from mkm.model_inputs import build_model_input_arrays, build_model_point_inputs
    from mkm.models.agpd_physics import parse_agpd_physics
    from mkm.models.agpd_registry import get_agpd_model_definition
    from mkm.workflows.agpd_basic import build_agpd_model_data

    output = (paths.agpd_posterior_root / "individual" / args.material / args.model / "posterior.nc"
              if args.source == "posterior" else
              paths.agpd_prior_predictive_root / "individual" / args.material / args.model / "prior_predictive.nc")
    draw_ids = args.draws if args.draws is not None else [args.draw]
    draws = {draw: saved_parameters(output, args.model, args.chain, draw, args.source) for draw in draw_ids}
    print(f"Saved parameter source: {output}; chain={args.chain}, draws={draw_ids}")
    for draw, params in draws.items():
        print(f"  Draw {draw} physical parameters: " + ", ".join(f"{k}={v:.6g}" for k, v in params.items()))
    model_data = elapsed("Load processed data", lambda: build_agpd_model_data(paths, (args.material,)))
    inputs = build_model_point_inputs(build_model_input_arrays(model_data))
    state = build_agpd_point_state(inputs, config)
    definition = get_agpd_model_definition(args.model)
    physics = parse_agpd_physics(args.model)
    lateral = dict(config["lateral_interactions"])
    if args.solver == "fixed":
        lateral.update(adaptive_convergence=False, fixed_point_iterations=96)
    print(f"Model points={len(state.E_V_SHE)}, compile mode={args.backend}, solver={args.solver}, "
          f"gradient={args.gradient}, repetitions={args.repeats}")
    print("Solver settings: " + str(lateral))
    for requested in args.points:
        ids = _select_point_ids(requested, len(state.E_V_SHE))
        count = len(ids)
        selected = replace(state, **{name: np.asarray(getattr(state, name))[ids]
                                     for name in ("E_V_SHE", "ln_a_CO", "ln_a_OH", "Pd_fraction",
                                                  "Ag_fraction", "material_index")},
                           theta_CO_max=(None if state.theta_CO_max is None else state.theta_CO_max[ids]))
        symbols = {field.name: pt.scalar(field.name, dtype="float64") for field in fields(definition.parameter_class)}
        parameters = definition.parameter_class(**symbols)
        result = evaluate_agpd_lateral_interactions(
            selected, parameters, config["temperature_K"], pathways=physics.pathways,
            finite_steps=physics.finite_steps, interactions=physics.interactions, lateral_config=lateral,
        )
        order = list(symbols)
        outputs = [result.ln_rate, *(result.pointwise[key] for key in (
            "coverage_solver_iterations", "coverage_solver_relative_residual",
            "coverage_branch_disagreement", "theta_CO"))]
        print(f"\n[Coverage points={count}, requested={requested}, IDs={ids[:20].tolist()}]", flush=True)
        function = elapsed("  Compile forward", lambda: pytensor.function(
            [symbols[key] for key in order], outputs, mode=args.backend, on_unused_input="ignore"))
        gradient_fn = None
        if args.gradient:
            print("  Constructing full physical-parameter gradient graph...", flush=True)
            gradient = elapsed("  Build gradient", lambda: pt.grad(pt.sum(result.ln_rate),
                                                                    [symbols[key] for key in order]))
            gradient_fn = elapsed("  Compile gradient", lambda: pytensor.function(
                [symbols[key] for key in order], gradient, mode=args.backend, on_unused_input="ignore"))
        print("  NOTE: each Scan stops globally; the iteration count is shared by every included model point.")
        for draw, params in draws.items():
            arguments = [params[key] for key in order]
            print(f"  --- Saved draw {args.chain}:{draw} ---", flush=True)
            first = elapsed("  Forward first call", lambda: function(*arguments))
            _summarize_coverages(first, "Forward check")
            durations = []
            for _ in range(args.repeats):
                t0 = perf_counter()
                function(*arguments)
                durations.append(perf_counter()-t0)
            print(f"  Forward warmed [median,min,max]: {np.median(durations):.5g}, "
                  f"{min(durations):.5g}, {max(durations):.5g} s", flush=True)
            if gradient_fn is not None:
                first_grad = elapsed("  Gradient first call", lambda: gradient_fn(*arguments))
                n_finite = sum(np.all(np.isfinite(g)) for g in first_grad)
                print(f"  Gradient components finite: {n_finite}/{len(first_grad)}")
                times = []
                for _ in range(args.repeats):
                    t0 = perf_counter()
                    gradient_fn(*arguments)
                    times.append(perf_counter()-t0)
                print(f"  Gradient warmed [median,min,max]: {np.median(times):.5g}, "
                      f"{min(times):.5g}, {max(times):.5g} s", flush=True)


def profile_logp(paths, config, args):
    from mkm.workflows.agpd_basic import build_agpd_model_data
    from mkm.workflows.agpd_fit import build_agpd_fit_model, resolve_agpd_fit_specification

    specification = resolve_agpd_fit_specification(config, model_name=args.model, material=args.material)
    data = elapsed("Load processed data", lambda: build_agpd_model_data(paths, (args.material,)))
    fit = elapsed("Build PyMC fit model", lambda: build_agpd_fit_model(specification, data, config))
    model = fit.built_model.model
    print("PyMC free RVs: " + ", ".join(v.name for v in model.free_RVs))
    point = model.initial_point()
    print("Transformed initial-point names: " + ", ".join(point))
    mode = args.backend
    print(f"Backend={mode}, repetitions={args.repeats}, evaluation at model.initial_point() "
          "(NOT a representative posterior draw)")
    logp = elapsed("Compile model logp", lambda: model.compile_logp(mode=mode))
    value = elapsed("Logp first call", lambda: logp(point))
    print(f"logp={float(np.asarray(value)):.8g}")
    durations = []
    for _ in range(args.repeats):
        t0 = perf_counter()
        logp(point)
        durations.append(perf_counter()-t0)
    print(f"Logp warmed [median,min,max]: {np.median(durations):.5g}, "
          f"{min(durations):.5g}, {max(durations):.5g} s")
    print("Constructing and compiling FULL joint logp gradient...", flush=True)
    dlogp = elapsed("Compile model dlogp", lambda: model.compile_dlogp(mode=mode))
    grad = elapsed("Dlogp first call", lambda: dlogp(point))
    g = np.asarray(grad, dtype=float)
    print(f"Gradient length={g.size}; finite={int(np.isfinite(g).sum())}/{g.size}; "
          f"L2 norm={float(np.linalg.norm(g)):.6g}")
    durations = []
    for _ in range(args.repeats):
        t0 = perf_counter()
        dlogp(point)
        durations.append(perf_counter()-t0)
    print(f"Dlogp warmed [median,min,max]: {np.median(durations):.5g}, "
          f"{min(durations):.5g}, {max(durations):.5g} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stats", "coverage", "logp"))
    parser.add_argument("--model", default="c_ER")
    parser.add_argument("--material", default="Ag50Pd50")
    parser.add_argument("--source", choices=("posterior", "prior"), default="posterior")
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--draw", type=int, default=50)
    parser.add_argument("--draws", nargs="+", type=int, help="Profile several saved draws with one compiled graph")
    parser.add_argument("--solver", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--backend", choices=("FAST_RUN", "NUMBA"), default="FAST_RUN")
    parser.add_argument("--points", nargs="+", type=int, default=(1, 16, 492))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gradient", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(p < 1 for p in args.points) or (args.draws is not None and not args.draws):
        parser.error("Repetitions and point counts must be positive")
    print(f"\n=== {args.stage.upper()} {args.model} / {args.material} ===", flush=True)
    print(f"Python={sys.version.split()[0]} platform={platform.platform()}")
    from mkm.project_paths import ProjectPaths
    from mkm.workflows.agpd_basic import load_agpd_model_config
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    if args.stage == "stats":
        output = paths.agpd_posterior_root / "individual" / args.material / args.model / "posterior.nc"
        print_stats(output)
    elif args.stage == "coverage":
        profile_coverage(paths, config, args)
    else:
        profile_logp(paths, config, args)
    print(f"=== {args.stage.upper()} COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
