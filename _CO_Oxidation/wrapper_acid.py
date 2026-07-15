from _CO_Oxidation.context import build_context

_CTX = build_context(env_type='acid')

process_experimental_data = _CTX.process_experimental_data
add_post_sampling_observables = _CTX.add_post_sampling_observables
calculate_flattened_r2 = _CTX.calculate_flattened_r2
plot_posteriors = _CTX.plot_posteriors
plot_model_fits = _CTX.plot_model_fits
plot_coverages = _CTX.plot_coverages
plot_drc = _CTX.plot_drc
simulate_fake_data = _CTX.simulate_fake_data

def __getattr__(name):
    return getattr(_CTX, name)
