"""Optional DRC and composition products for AgPd posterior postprocessing."""

from mkm.model_inputs import build_model_point_inputs
from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
    plot_agpd_composition_parameter_overview,
)
from mkm.postprocessing.drc import (
    compare_transition_state_drc_steps,
    compute_composition_transition_state_drc,
    compute_transition_state_drc,
)
from mkm.postprocessing.plotting import plot_transition_state_drc


def _compute_drc(run, config, step_eV):
    point_inputs = build_model_point_inputs(run.inputs)
    kwargs = {
        "inference_data": run.inference_data,
        "model_name": run.specification.model_name,
        "point_inputs": point_inputs,
        "model_points": run.model_data.model_points,
        "config": config,
        "step_eV": step_eV,
    }
    if run.specification.is_all_materials:
        return compute_composition_transition_state_drc(
            **kwargs, parameterization=run.specification.parameterization
        )
    return compute_transition_state_drc(**kwargs)


def postprocess_agpd_drc(
    run, config, paths, *, step_eV=1e-4, check_half_step=False, save_draws=False, make_plots=True
):
    drc_dir = paths.fit_drc_dir(run.output_dir)
    drc_dir.mkdir(parents=True, exist_ok=True)
    result = _compute_drc(run, config, step_eV)
    result.summary.to_parquet(drc_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(drc_dir / "drc_transition_state_checks.csv", index=False)
    if save_draws:
        result.draws.to_netcdf(drc_dir / "drc_transition_state_draws.nc")
    if check_half_step:
        half_step = _compute_drc(run, config, 0.5 * step_eV)
        compare_transition_state_drc_steps(result, half_step).to_csv(
            drc_dir / "drc_transition_state_step_convergence.csv", index=False
        )
    if make_plots:
        figures_dir = paths.fit_figures_dir(run.output_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        for material in run.inputs.materials:
            summary = result.summary.loc[result.summary["material"] == material]
            if summary.empty:
                continue
            material_dir = figures_dir / material
            material_dir.mkdir(parents=True, exist_ok=True)
            plot_transition_state_drc(
                summary,
                f"{run.specification.model_name}, {material}",
                material_dir / "drc_transition_states.png",
            )
    return result


def postprocess_agpd_composition(run, config, paths, *, n_grid=181):
    if not run.specification.is_all_materials:
        return None
    composition_dir = paths.fit_composition_dir(run.output_dir)
    composition_dir.mkdir(parents=True, exist_ok=True)
    trends = build_agpd_composition_parameter_trends(
        run.inference_data,
        config,
        model_name=run.specification.model_name,
        parameterization=run.specification.parameterization,
        n_grid=n_grid,
        prior_material=run.specification.prior_material or "Ag10Pd90",
        materials=run.inputs.materials,
    )
    trends_path = composition_dir / "composition_parameter_trends.parquet"
    trends.to_parquet(trends_path, index=False)
    figure_path = composition_dir / "composition_parameter_overview.png"
    plot_agpd_composition_parameter_overview(
        trends,
        config,
        model_name=run.specification.model_name,
        parameterization=run.specification.parameterization,
        output_path=figure_path,
        error_structure=run.specification.error_structure,
        materials=run.inputs.materials,
    )
    return trends
