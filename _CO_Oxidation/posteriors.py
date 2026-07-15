# posteriors.py
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pymc as pm

class PosteriorsMixin:
    def plot_posteriors(self, trace, model):
        delta_name = self.cfg['delta_name']
        all_vars = list(trace.posterior.data_vars)
        excluded_vars = ['alpha', delta_name, 'delta_CO', 'rate', 'log_rate', 'log_residual']
        var_names = [v for v in all_vars if not v.endswith("__")]
        kinetic_vars = [v for v in var_names if not v.startswith(('theta', 'phi')) and v not in excluded_vars]
        summary = az.summary(trace, var_names=kinetic_vars)
        print(summary)

        if not hasattr(trace, "prior"):
            with model:
                valid_prior_vars = [v for v in kinetic_vars if v in model.named_vars]
                prior_trace = pm.sample_prior_predictive(samples=10000, var_names=valid_prior_vars)
                trace.add_groups({"prior": prior_trace.prior})

        print("\n--- Prior-to-Posterior Contraction ---")
        for var in kinetic_vars:
            if var in trace.prior.data_vars:
                prior_var = np.var(trace.prior[var].values)
                post_var = np.var(trace.posterior[var].values)
                contraction = 1.0 - (post_var / prior_var)
                print(f"{var}: {contraction:.3f}")

        num_vars = len(kinetic_vars)
        cols = min(4, max(1, num_vars))
        rows = int(np.ceil(num_vars / cols))
        rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 1.5}
        figsize_post = (2.5 * cols, 3.0 * rows)

        with plt.rc_context(rc_update):
            if kinetic_vars:
                axes = az.plot_posterior(trace, var_names=kinetic_vars, hdi_prob=0.95, round_to=3, figsize=figsize_post, grid=(rows, cols), textsize=10)
                axes_flat = np.atleast_1d(axes).flatten()
                for ax, var_name in zip(axes_flat, kinetic_vars):
                    for spine in ['top', 'left', 'right']:
                        ax.spines[spine].set_visible(False)
                    if hasattr(trace, 'prior') and var_name in trace.prior.data_vars:
                        orig_xlim = ax.get_xlim()
                        ax_prior = ax.twinx()
                        prior_samples = trace.prior[var_name].values.flatten()
                        az.plot_dist(prior_samples, ax=ax_prior, color='gray', plot_kwargs={'alpha': 0.8}, fill_kwargs={'alpha': 0.15})
                        lines = ax_prior.get_lines()
                        if lines:
                            x_data = lines[0].get_xdata()
                            y_data = lines[0].get_ydata()
                            mask = (x_data >= orig_xlim[0]) & (x_data <= orig_xlim[1])
                            if np.any(mask):
                                y_max_visible = np.max(y_data[mask])
                                if y_max_visible > 0:
                                    ax_prior.set_ylim(0, y_max_visible * 1.15) 
                        ax_prior.set_yticks([])
                        for spine in ax_prior.spines.values():
                            spine.set_visible(False)
                        ax.set_xlim(orig_xlim)
                        ax.set_zorder(ax_prior.get_zorder() + 1)
                        ax.patch.set_visible(False) 

                    if ax is not None and hasattr(ax, 'texts'):
                        for text_obj in ax.texts:
                            if 'mean' in text_obj.get_text():
                                text_obj.set_fontweight('bold')
                        ax.title.set_fontweight('bold')
                        
                plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.88)
                plt.show()

            if len(kinetic_vars) > 1:
                plot_pair_vars = [v for v in kinetic_vars if v not in ['CO_converge_error', 'OH_converge_error', 'sigma_base', 'sigma_rel', 'exponent']]
                az.plot_pair(trace, var_names=plot_pair_vars, kind='kde', divergences=True, figsize=(8, 8), textsize=10)
                plt.show()

        with model:
            ppc = pm.sample_posterior_predictive(trace, progressbar=False, extend_inferencedata=True)
        for var_name in ppc.observed_data.data_vars:
            r2 = self.calculate_flattened_r2(ppc, var_name)
            print(f"{var_name}: {r2:.3f}")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            az.plot_loo_pit(idata=trace, y=var_name, ecdf=False, color='blue', ax=axes[0], hdi_prob=0.95)
            axes[0].set_title(f"LOO-PIT (KDE Density): {var_name}", fontweight='bold')
            az.plot_loo_pit(idata=trace, y=var_name, ecdf=True, color='blue', ax=axes[1], hdi_prob=0.95)
            axes[1].set_title(f"LOO-PIT (ECDF Difference): {var_name}", fontweight='bold')
            az.plot_energy(trace, ax=axes[2])
            axes[2].set_title("Energy", fontweight='bold')
            plt.tight_layout()
            plt.show()

        return ppc