"""PtRu transition-state DRC output workflow."""

from mkm.model_inputs import build_model_point_inputs
from mkm.postprocessing.drc import compute_ptru_transition_state_drc, compare_transition_state_drc_steps
from mkm.postprocessing.drc_status import record_drc_status
from mkm.postprocessing.plotting.drc import plot_transition_state_drc
from mkm.workflows._ptru_postprocess_products import ptru_electrolyte_label


def postprocess_ptru_drc(
    run, paths, *, temperature_K, step_eV=1e-4, check_half_step=False,
    save_draws=False, make_plots=True,
):
    """Write PtRu TS-DRC outputs, including for pooled composition fits."""
    from mkm.workflows.ptru_fit import load_ptru_config

    config, priors = load_ptru_config(paths)
    point_inputs = build_model_point_inputs(run.inputs)

    def compute(step):
        return compute_ptru_transition_state_drc(
            run.inference_data, run.specification, point_inputs, run.model_data.model_points,
            config, priors, step_eV=step,
        )

    drc_dir = paths.fit_drc_dir(run.output_dir)
    drc_dir.mkdir(parents=True, exist_ok=True)
    for optional in ("drc_transition_state_step_convergence.csv", "drc_transition_state_draws.nc"):
        (drc_dir / optional).unlink(missing_ok=True)
    result = compute(step_eV)
    result.summary.to_parquet(drc_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(drc_dir / "drc_transition_state_checks.csv", index=False)
    if save_draws:
        result.draws.to_netcdf(drc_dir / "drc_transition_state_draws.nc")
    if check_half_step:
        compare_transition_state_drc_steps(result, compute(step_eV / 2)).to_csv(
            drc_dir / "drc_transition_state_step_convergence.csv", index=False,
        )
    if make_plots:
        for material in run.inputs.materials:
            figures_dir = paths.fit_figures_dir(run.output_dir) / material
            figures_dir.mkdir(parents=True, exist_ok=True)
            summary = result.summary.loc[result.summary["material"] == material]
            plot_transition_state_drc(
                summary, f"{run.specification.model_name}, {material} / {run.specification.medium}",
                figures_dir / "drc_transition_states.png",
                electrolyte_label=ptru_electrolyte_label(run.specification.medium),
            )
    record_drc_status(drc_dir, recomputed=True, figures_dir=paths.fit_figures_dir(run.output_dir), options={
        "step_eV": step_eV, "check_half_step": check_half_step,
        "save_draws": save_draws, "make_plots": make_plots,
    })
    return result


# Compatibility name for historical direct DRC consumers.
postprocess_pt100_drc = postprocess_ptru_drc
