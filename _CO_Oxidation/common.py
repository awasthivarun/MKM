import arviz as az; import matplotlib.pyplot as plt
import numpy as np; import pandas as pd; import pymc as pm
import pytensor; import xarray as xr
pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); pd.options.future.infer_string = False
pd.set_option('display.max_colwidth', None); pd.options.mode.string_storage = "python"

R, T, F = 8.3145, 293.15, 96485
kb_eV, h, kb_J = 8.617e-5, 6.626e-34, 1.3806e-23
N_A = 6.022e23

def _resolve_config(env_type):
    is_acid = env_type.lower() == 'acid'
    if is_acid:
        return {
            'concentration_list_name': 'C_H_list',
            'concentration_label': 'M HClO$_4$',
            'delta_name': 'delta_H',
            'delta_exp_name': 'delta_H_exp',
            'delta_sd_name': 'delta_H_SD_exp',
            'delta_E_name': 'E_H_exp',
            'delta_data_key': 'delta_H',
            'delta_sd_data_key': 'delta_H_SD',
            'delta_E_data_key': 'E_H',
            'delta_ylabel': 'Order (H+)',
            'delta_title': 'Proton Reaction Order',
        }
    return {
        'concentration_list_name': 'C_KOH_list',
        'concentration_label': 'M KOH',
        'delta_name': 'delta_OH',
        'delta_exp_name': 'delta_OH_exp',
        'delta_sd_name': 'delta_OH_SD_exp',
        'delta_E_name': 'E_OH_exp',
        'delta_data_key': 'delta_OH',
        'delta_sd_data_key': 'delta_OH_SD',
        'delta_E_data_key': 'E_OH',
        'delta_ylabel': 'Order (OH)',
        'delta_title': 'OH Reaction Order',
    }

def _axes_to_2d(axes, nrows, ncols):
    if isinstance(axes, np.ndarray):
        if axes.ndim == 2:
            return axes
        if nrows == 1:
            return axes.reshape(1, -1)
        if ncols == 1:
            return axes.reshape(-1, 1)
        return np.atleast_2d(axes)
    return np.array([[axes]], dtype=object)

class PlottingContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {
            'truncated_E_exp': {},
            'truncated_rate_exp': {},
            'truncated_rate_SD_exp': {},
            'truncated_log_rate_exp': {},
            'truncated_log_rate_SD_exp': {},
            'truncated_rate_matrix': {},
            'E_exp_map': {},
            'rate_exp': {},
            'rate_SD_exp': {},
            'log_rate_exp': {},
            'log_rate_SD_exp': {},
            'alpha_exp': {},
            'alpha_SD_exp': {},
            'E_H_exp': {},
            'delta_H_exp': {},
            'delta_H_SD_exp': {},
            'E_OH_exp': {},
            'delta_OH_exp': {},
            'delta_OH_SD_exp': {},
            'E_CO_exp': {},
            'delta_CO_exp': {},
            'delta_CO_SD_exp': {},
            'C_H_list': None,
            'C_KOH_list': None,
            'P_CO_list': None,
            'E_in': None,
            'C_H_in': None,
            'C_KOH_in': None,
            'P_CO_in': None,
            'rate_obs': None,
            'rate_SD_obs': None,
            'log_rate_obs': None,
            'log_rate_SD_obs': None,
            'rate_obs_matrix': None,
        }

    def __getattr__(self, name):
        if name in self.state:
            return self.state[name]
        raise AttributeError(name)

    def process_experimental_data(self, experiments, C_list, P_list):
        list_name = self.cfg['concentration_list_name']
        self.state[list_name] = C_list
        self.state['P_CO_list'] = P_list
        
        E_in, C_H_in, C_KOH_in, P_CO_in = [], [], [], []
        rate_obs, rate_SD_obs, log_rate_obs, log_rate_SD_obs = [], [], [], []
        rate_obs_matrix = []

        for C in C_list:
            for P in P_list:
                cond = (C, P)
                data = experiments[cond]
                self.state['truncated_E_exp'][cond] = data['truncated_E']
                self.state['truncated_rate_exp'][cond] = data['truncated_rate']
                self.state['truncated_rate_SD_exp'][cond] = data['truncated_rate_SD']
                self.state['truncated_log_rate_exp'][cond] = data['truncated_log_rate']
                self.state['truncated_log_rate_SD_exp'][cond] = data['truncated_log_rate_SD']
                self.state['truncated_rate_matrix'][cond] = data['truncated_rate_matrix']
                self.state['E_exp_map'][cond] = data['E']
                self.state['rate_exp'][cond] = data['rate']
                self.state['rate_SD_exp'][cond] = data['rate_SD']
                self.state['log_rate_exp'][cond] = data['log_rate']
                self.state['log_rate_SD_exp'][cond] = data['log_rate_SD']
                self.state['alpha_exp'][cond] = data['alpha']
                self.state['alpha_SD_exp'][cond] = data['alpha_SD']
                self.state[self.cfg['delta_E_name']][cond] = data[self.cfg['delta_E_data_key']]
                self.state[self.cfg['delta_exp_name']][cond] = data[self.cfg['delta_data_key']]
                self.state[self.cfg['delta_sd_name']][cond] = data[self.cfg['delta_sd_data_key']]
                if 'E_CO' in data:
                    self.state['E_CO_exp'][cond] = data['E_CO']
                    self.state['delta_CO_exp'][cond] = data['delta_CO']
                    self.state['delta_CO_SD_exp'][cond] = data['delta_CO_SD']
                
                E_arr = data['truncated_E']
                E_in.append(E_arr)
                C_H_in.append(np.full_like(E_arr, C))
                C_KOH_in.append(np.full_like(E_arr, C))
                P_CO_in.append(np.full_like(E_arr, P))
                rate_obs.append(data['truncated_rate'])
                rate_SD_obs.append(data['truncated_rate_SD'])
                log_rate_obs.append(data['truncated_log_rate'])
                log_rate_SD_obs.append(data['truncated_log_rate_SD'])
                rate_obs_matrix.append(data['truncated_rate_matrix'])

        self.state['E_in'] = np.concatenate(E_in)
        self.state['C_H_in'] = np.concatenate(C_H_in)
        self.state['C_KOH_in'] = np.concatenate(C_KOH_in)
        self.state['P_CO_in'] = np.concatenate(P_CO_in)
        self.state['rate_obs'] = np.concatenate(rate_obs)
        self.state['rate_SD_obs'] = np.concatenate(rate_SD_obs)
        self.state['log_rate_obs'] = np.concatenate(log_rate_obs)
        self.state['log_rate_SD_obs'] = np.concatenate(log_rate_SD_obs)
        self.state['rate_obs_matrix'] = np.concatenate(rate_obs_matrix, axis=1)

    def add_post_sampling_observables(self, trace):
        conc_list = self.state[self.cfg['concentration_list_name']]
        p_list = self.state['P_CO_list']

        log_rate_samples = trace.posterior['log_rate'].values
        alpha_samples = np.zeros_like(log_rate_samples)
        model_log_rates = {}
        start = 0
        for C in conc_list:
            for P in p_list:
                E_arr = self.state['truncated_E_exp'][(C, P)]
                end = start + len(E_arr)
                alpha_samples[:, :, start:end] = (R * T / F) * np.gradient(log_rate_samples[:, :, start:end], E_arr, axis=2)
                model_log_rates[(C, P)] = log_rate_samples[:, :, start:end]
                start = end
        trace.posterior['alpha'] = (trace.posterior['log_rate'].dims, alpha_samples)

        delta_name = self.cfg['delta_name']
        delta_list = []
        ln_C = np.log(conc_list)
        w_C = ((ln_C - np.mean(ln_C)) / np.sum((ln_C - np.mean(ln_C)) ** 2))[None, None, :, None]
        for P in p_list:
            E_arrays = [np.round(self.state['truncated_E_exp'][(C, P)], 6) for C in conc_list]
            common_E = E_arrays[0]
            for arr in E_arrays[1:]:
                common_E = np.intersect1d(common_E, arr)
            if common_E.size == 0:
                raise ValueError(f'No common energies found for {delta_name} at P_CO={P}.')
            rates_for_fit = []
            for C in conc_list:
                E_arr = np.round(self.state['truncated_E_exp'][(C, P)], 6)
                mask = np.isin(E_arr, common_E)
                rates_for_fit.append(model_log_rates[(C, P)][:, :, mask])
            rates_for_fit = np.stack(rates_for_fit, axis=2)
            delta_vals = np.sum(w_C * rates_for_fit, axis=2)
            delta_list.append(delta_vals)
        trace.posterior[delta_name] = (("chain", "draw", f'{delta_name}_flat'), np.concatenate(delta_list, axis=2))

        delta_co_list = []
        for C in conc_list:
            for i in range(len(p_list) - 1):
                P1, P2 = p_list[i], p_list[i + 1]
                E1 = np.round(self.state['truncated_E_exp'][(C, P1)], 6)
                E2 = np.round(self.state['truncated_E_exp'][(C, P2)], 6)
                common_E = np.intersect1d(E1, E2)
                if common_E.size == 0:
                    raise ValueError(f'No common energies found for delta_CO at C={C}, P={P1}/{P2}.')
                mask1 = np.isin(E1, common_E)
                mask2 = np.isin(E2, common_E)
                lr1 = model_log_rates[(C, P1)][:, :, mask1]
                lr2 = model_log_rates[(C, P2)][:, :, mask2]
                delta_co = (lr2 - lr1) / np.log(P2 / P1)
                delta_co_list.append(delta_co)
        trace.posterior['delta_CO'] = (("chain", "draw", 'delta_CO_flat'), np.concatenate(delta_co_list, axis=2))
        return trace

    @staticmethod
    def calculate_flattened_r2(ppc, var_name):
        y_true = ppc.observed_data[var_name].values
        y_pred = ppc.posterior_predictive[var_name].values
        chains, draws = y_pred.shape[:2]
        y_pred_flat = y_pred.reshape(chains * draws, -1)
        y_true_flat = y_true.flatten()
        return az.r2_score(y_true_flat, y_pred_flat)['r2']

    def plot_posteriors(self, trace, model):
        delta_name = self.cfg['delta_name']
        all_vars = list(trace.posterior.data_vars)
        excluded_vars = ['alpha', delta_name, 'delta_CO', 'rate', 'log_rate']
        var_names = [v for v in all_vars if not v.endswith("__")]
        kinetic_vars = [v for v in var_names if not v.startswith(('theta', 'phi')) and v not in excluded_vars]
        summary = az.summary(trace, var_names=kinetic_vars)
        print(summary)

        num_vars = len(kinetic_vars)
        cols = min(4, max(1, num_vars))
        rows = int(np.ceil(num_vars / cols))
        rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 1.5}
        figsize_post = (2.5 * cols, 3.0 * rows)

        with plt.rc_context(rc_update):
            if kinetic_vars:
                axes = az.plot_posterior(trace, var_names=kinetic_vars, hdi_prob=0.95, round_to=3, figsize=figsize_post, grid=(rows, cols), textsize=10)
                axes_flat = np.array(axes).flatten()
                for ax in axes_flat:
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
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
        for var_name in ppc.observed_data.data_vars:
            r2 = self.calculate_flattened_r2(ppc, var_name)
            print(f"{var_name}: {r2:.3f}")
        return ppc

    def plot_model_fits(self, trace, ppc, consolidated=False):
        conc_list = self.state[self.cfg['concentration_list_name']]
        p_list = self.state['P_CO_list']
        delta_name = self.cfg['delta_name']

        n_C = len(conc_list)
        n_P = len(p_list)
        figsize_main = (1.5 + 1.5 * n_C, 1.0 + 1.5 * n_P)
        figsize_CO = (1.5 + 1.5 * n_C, 1.0 + 1.5 * max(1, n_P - 1))
        figsize_OH = (1.5 + 1.5 * n_P, 1.0 + 1.5 * 1)
        figsize_1x1 = (4.0, 3.6)
        figsize_cons = (1.5 + 1.5 * n_C, 1.0 + 1.5 * 1)
        title_size, label_size = 12, 10

        colors = plt.cm.tab10.colors[:len(p_list)]
        ci_multiplier = 1
        ppc_vars = [v for v in ppc.observed_data.data_vars if not v.endswith("__")]
        if len(ppc_vars) != 1:
            raise ValueError(f'Expected exactly one PPC observed variable, found {ppc_vars}')
        ppc_var = ppc_vars[0]
        non_ppc_var = 'log_rate' if ppc_var == 'rate' else 'rate'

        def get_model_slice(var_fit, C_val, P_val, is_ppc=False):
            dataset = ppc.posterior_predictive if is_ppc else trace.posterior
            track_name = ppc_var if is_ppc else var_fit
            if var_fit in ['log_rate', 'alpha', 'rate'] or is_ppc:
                data_arr = dataset[track_name]
                if is_ppc and data_arr.ndim == 4:
                    if data_arr.shape[2] < data_arr.shape[3]:
                        data_arr = data_arr.isel(**{data_arr.dims[2]: 0})
                    else:
                        data_arr = data_arr.isel(**{data_arr.dims[3]: 0})
                idx = 0
                for C in conc_list:
                    for P in p_list:
                        length = len(self.state['truncated_E_exp'][(C, P)])
                        if C == C_val and P == P_val:
                            return data_arr[:, :, idx:idx + length], self.state['truncated_E_exp'][(C, P)]
                        idx += length
            elif var_fit == delta_name:
                idx = 0
                for P in p_list:
                    E_arrays = [np.round(self.state['truncated_E_exp'][(C, P)], 6) for C in conc_list]
                    common_E = E_arrays[0]
                    for arr in E_arrays[1:]:
                        common_E = np.intersect1d(common_E, arr)
                    length = len(common_E)
                    if P == P_val:
                        return dataset[var_fit][:, :, idx:idx + length], common_E
                    idx += length
            elif var_fit == 'delta_CO':
                idx = 0
                for C in conc_list:
                    for i in range(len(p_list) - 1):
                        P1, P2 = p_list[i], p_list[i + 1]
                        E1 = np.round(self.state['truncated_E_exp'][(C, P1)], 6)
                        E2 = np.round(self.state['truncated_E_exp'][(C, P2)], 6)
                        common_E = np.intersect1d(E1, E2)
                        length = len(common_E)
                        if C == C_val and P1 == P_val:
                            return dataset[var_fit][:, :, idx:idx + length], common_E
                        idx += length
            raise ValueError(f'No matching data slice found for {var_fit} at C={C_val}, P={P_val}.')

        def plot_grid(var_fit, nrows, ncols, figsize, ylabel, title_text, consolidated=False):
            is_ppc = (var_fit == ppc_var)
            share_y = False if is_ppc else True
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex='col', sharey=share_y)
            axes_2d = _axes_to_2d(axes, nrows, ncols)
            fig.suptitle(title_text, fontsize=title_size, fontweight='bold')
            fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size + 2)
            fig.supylabel(ylabel, fontweight='bold', fontsize=label_size + 2)

            if consolidated and var_fit in ['alpha', 'delta_CO']:
                legend_handles = []
                linestyles = [':', (0, (1, 0.5)), '--', '-']
                ax_list = axes_2d.flatten()
                for i, C_val in enumerate(conc_list):
                    ax = ax_list[i]
                    loop_limit = len(p_list) if var_fit != 'delta_CO' else len(p_list) - 1
                    for pressure_idx in range(loop_limit):
                        P1 = p_list[pressure_idx]
                        P2 = p_list[pressure_idx + 1] if var_fit == 'delta_CO' else None
                        cond_key = (C_val, P1)
                        if var_fit == 'alpha':
                            exp_mean = self.state['alpha_exp'][cond_key]
                            E_exp = self.state['E_exp_map'][cond_key]
                        else:
                            exp_mean = self.state['delta_CO_exp'][cond_key]
                            E_exp = self.state['E_CO_exp'][cond_key]
                        model_slice, E_model = get_model_slice(var_fit, C_val, P1, is_ppc=is_ppc)
                        model_mean = model_slice.mean(dim=("chain", "draw"))
                        hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                        label_str = f'{P1} atm' if var_fit != 'delta_CO' else f'{P1} - {P2} atm'
                        line, = ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5, label=label_str)
                        ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                        ax.plot(E_exp, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.20, zorder=2)
                        if i == 0: legend_handles.append(line)
                    ax.text(0.05, 0.95, f'{C_val} {self.cfg["concentration_label"]}', transform=ax.transAxes, fontsize=label_size, fontweight='bold', va='top', ha='left')
                    for label in ax.get_yticklabels(): label.set_fontweight('bold')
                    for label in ax.get_xticklabels(): label.set_fontweight('bold')
                ymin, ymax = ax_list[0].get_ylim()
                y_margin = (ymax - ymin) * 0.10
                ax_list[0].set_ylim(ymin, ymax + y_margin)
                plt.tight_layout()
                plt.subplots_adjust(right=0.98, top=0.90, left=0.10, bottom=0.27, wspace=0.1)
                fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(legend_handles), frameon=False, prop={'size': label_size, 'weight': 'bold'})
                plt.show()
                return

            if consolidated and var_fit == delta_name:
                legend_handles = []
                linestyles = [':', (0, (1, 0.5)), '--', '-']
                ax = axes_2d[0, 0]
                C_val = conc_list[0]
                for pressure_idx, P_val in enumerate(p_list):
                    cond_key = (C_val, P_val)
                    exp_mean = self.state[self.cfg['delta_exp_name']][cond_key]
                    E_exp = self.state[self.cfg['delta_E_name']][cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_val, P_val, is_ppc=is_ppc)
                    model_mean = model_slice.mean(dim=("chain", "draw"))
                    hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                    line, = ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5, label=f'{P_val} atm')
                    ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                    ax.plot(E_exp, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.20, zorder=2)
                    legend_handles.append(line)
                for label in ax.get_yticklabels(): label.set_fontweight('bold')
                for label in ax.get_xticklabels(): label.set_fontweight('bold')
                plt.tight_layout()
                plt.subplots_adjust(right=0.98, top=0.92, left=0.2, bottom=0.25)
                fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=len(legend_handles)/2, frameon=False, prop={'size': label_size, 'weight': 'bold'})
                plt.show()
                return

            var_map = {
                'log_rate': ('log_rate_exp', 'log_rate_SD_exp', 'E_exp_map'),
                'alpha':    ('alpha_exp', 'alpha_SD_exp', 'E_exp_map'),
                'rate':     ('rate_exp', 'rate_SD_exp', 'E_exp_map'),
                'delta_CO': ('delta_CO_exp', 'delta_CO_SD_exp', 'E_CO_exp'),
                delta_name: (self.cfg['delta_exp_name'], self.cfg['delta_sd_name'], self.cfg['delta_E_name'])
            }

            for i, C_val in enumerate(conc_list):
                if var_fit == self.cfg['delta_name'] and i > 0:
                    continue
                loop_p_list = p_list[:-1] if var_fit == 'delta_CO' else p_list
                for j, P_val in enumerate(loop_p_list):
                    if var_fit == self.cfg['delta_name']:
                        ax = axes_2d[0, j]; C_lookup = conc_list[0]
                    else:
                        ax = axes_2d[j, i]; C_lookup = C_val
                    cond_key = (C_lookup, P_val)

                    mean_key, sd_key, E_key = var_map[var_fit]
                    exp_mean = self.state[mean_key][cond_key]
                    exp_sd = self.state[sd_key][cond_key]
                    E_exp = self.state[E_key][cond_key]

                    model_slice, E_model = get_model_slice(var_fit, C_val, P_val, is_ppc=is_ppc)
                    model_mean = model_slice.mean(dim=("chain", "draw"))
                    hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                    hdi_90 = az.hdi(model_slice, hdi_prob=0.90)[model_slice.name]
                    if not consolidated:
                        exp_ci = exp_sd * ci_multiplier
                        ax.fill_between(E_exp, exp_mean - exp_ci, exp_mean + exp_ci, color='gray', alpha=0.3, linewidth=0, zorder=1)
                    ax.plot(E_exp, exp_mean, color='black', lw=2.5, alpha=0.6, zorder=2)
                    ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[j], alpha=0.15, linewidth=0, zorder=3)
                    ax.fill_between(E_model, hdi_90[:, 0].values, hdi_90[:, 1].values, color=colors[j], alpha=0.30, linewidth=0, zorder=4)
                    ax.plot(E_model, model_mean, color=colors[j], lw=3.2, zorder=5)
                    
                    is_OH_grid = (var_fit == self.cfg['delta_name'])
                    is_CO_grid = (var_fit == 'delta_CO')
                    if j == 0 and not is_OH_grid: 
                        ax.set_title(f'{C_val} {self.cfg["concentration_label"]}', fontsize=label_size, fontweight='bold')
                    if is_OH_grid: 
                        ax.set_title(f'{P_val} atm', fontsize=label_size, fontweight='bold')
                    if i == n_C - 1 and not is_OH_grid:
                        label_right = f'{P_val} atm' if not is_CO_grid else f'{p_list[j]} - {p_list[j+1]} atm'
                        ax.text(1.05, 0.5, label_right, transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')

            is_OH_grid = (var_fit == self.cfg['delta_name'])
            is_CO_grid = (var_fit == 'delta_CO')
            bottom_adj = 0.20 if is_OH_grid else (0.10 if is_CO_grid else 0.08)
            top_adj = 0.80 if is_OH_grid else (0.88 if is_CO_grid else 0.90)
            plt.tight_layout()
            plt.subplots_adjust(right=0.92, top=top_adj, left=0.10, bottom=bottom_adj)
            plt.show()

        rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2}
        with plt.rc_context(rc_update):
            plot_grid(ppc_var, n_P, n_C, figsize_main, "TOF (1/s)" if ppc_var == 'rate' else "log Rate", "Rate" if ppc_var == 'rate' else "Log Rate")
            plot_grid(non_ppc_var, n_P, n_C, figsize_main, "log Rate" if non_ppc_var == 'log_rate' else "TOF (1/s)", "Log Rate" if non_ppc_var == 'log_rate' else "Rate")
            if consolidated:
                plot_grid('alpha', 1, n_C, figsize_cons, 'alpha', 'Transfer Coefficients', consolidated=True)
                plot_grid(delta_name, 1, 1, figsize_1x1, self.cfg['delta_ylabel'], self.cfg['delta_title'], consolidated=True)
                plot_grid('delta_CO', 1, n_C, figsize_cons, 'Order (CO)', 'CO Reaction Order', consolidated=True)
            else:
                plot_grid('alpha', n_P, n_C, figsize_main, 'alpha', 'Transfer Coefficients')
                plot_grid(delta_name, 1, n_P, figsize_OH, self.cfg['delta_ylabel'], self.cfg['delta_title'])
                plot_grid('delta_CO', n_P - 1, n_C, figsize_CO, 'Order (CO)', 'CO Reaction Order')

    def plot_coverages(self, trace):
        conc_list = self.state[self.cfg['concentration_list_name']]
        p_list = self.state['P_CO_list']

        n_C = len(conc_list)
        n_P = len(p_list)
        figsize = (1.5 + 1.5 * n_C, 1.0 + 1.5 * n_P)
        title_size, label_size = 12, 10

        rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2}
        with plt.rc_context(rc_update):
            fig, axes = plt.subplots(nrows=n_P, ncols=n_C, figsize=figsize, sharex='col', sharey=True)
            axes_2d = _axes_to_2d(axes, n_P, n_C)
            fig.suptitle('Modeled Surface Coverages', fontsize=title_size, fontweight='bold', y=0.97)
            fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size + 2)
            fig.supylabel('Coverage', fontweight='bold', fontsize=label_size + 2)

            index_map = {}
            _idx = 0
            for C in conc_list:
                for P in p_list:
                    L = len(self.state['truncated_E_exp'][(C, P)])
                    index_map[(C, P)] = (_idx, _idx + L)
                    _idx += L

            cov_vars = ['theta_CO', 'theta_OH', 'theta_COOH', 'theta_empty', 'theta_OH_star', 'theta_OH_pound']
            cov_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:gray', 'tab:blue', 'tab:pink']
            cov_labels = [r'$\theta_{CO}$', r'$\theta_{OH}$', r'$\theta_{COOH}$', r'$\theta_{*}$', r'$\theta^*_{OH}$', r'$\theta^\#_{OH}$']

            for i, C_val in enumerate(conc_list):
                for j, P_val in enumerate(p_list):
                    ax = axes_2d[j, i]
                    start, end = index_map[(C_val, P_val)]
                    for v, col, lab in zip(cov_vars, cov_colors, cov_labels):
                        if v not in trace.posterior:
                            continue
                        data = trace.posterior[v][:, :, start:end]
                        mu = data.mean(dim=("chain", "draw"))
                        hdi_95 = az.hdi(data, hdi_prob=0.95)[data.name]
                        lower = hdi_95[:, 0].values
                        upper = hdi_95[:, 1].values
                        ax.fill_between(self.state['truncated_E_exp'][(C_val, P_val)], lower, upper, color=col, alpha=0.15, linewidth=0)
                        ax.plot(self.state['truncated_E_exp'][(C_val, P_val)], mu, color=col, label=lab, lw=2.5)
                    ax.set_ylim(-0.05, 1.05)
                    if j == 0:
                        ax.set_title(f'{C_val} {self.cfg["concentration_label"]}', fontsize=label_size, fontweight='bold')
                    if i == n_C - 1:
                        ax.text(1.05, 0.5, f'{P_val} atm', transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')
                    if i == 0 and j == 0:
                        ax.legend(loc='best', frameon=False, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(right=0.92, top=0.90, bottom=0.08, left=0.10)
            plt.show()

    def plot_drc(self, model, trace, perturb_vars, perturb_labels=None, var_types=None):
        conc_list = self.state[self.cfg['concentration_list_name']]
        p_list = self.state['P_CO_list']

        n_C = len(conc_list); n_P = len(p_list)
        figsize = (1.5 + 1.5 * n_C, 1.0 + 1.5 * n_P)
        title_size, label_size = 12, 10

        if var_types is None:
            var_types = {v: 'Gact' for v in perturb_vars}
        if perturb_labels is None:
            perturb_labels = [v.replace('_0', '') for v in perturb_vars]

        delta_rel = 0
        target_node = model['log_rate']
        replacements, sym_inputs, input_names = {}, [], []
        post = trace.posterior
        n_chains, n_draws = post.sizes['chain'], post.sizes['draw']
        total_samples = n_chains * n_draws

        for rv in model.free_RVs:
            if rv.name in post:
                base_shape = post[rv.name].shape[2:]
                if len(base_shape) == 0:
                    sym_var = pytensor.tensor.vector(rv.name)
                else:
                    sym_var = pytensor.tensor.tensor(rv.dtype, shape=(None,) + base_shape)
                sym_inputs.append(sym_var)
                replacements[rv] = sym_var
                input_names.append(rv.name)
                if rv.name in model.named_vars:
                    replacements[model[rv.name]] = sym_var

        for var_name in perturb_vars:
            if var_name in input_names:
                idx = input_names.index(var_name)
                replacements[model[var_name]] = sym_inputs[idx]

        from pytensor.graph import vectorize_graph
        cloned_node = vectorize_graph(target_node, replace=replacements)
        calc_fn = pytensor.function(sym_inputs, cloned_node, on_unused_input='ignore')

        base_inputs = []
        for name in input_names:
            val = post[name].values.reshape(total_samples, *post[name].shape[2:])
            base_inputs.append(val)

        X_dict = {}
        for i, var_name in enumerate(perturb_vars):
            var_idx = input_names.index(var_name)
            base_val_array = base_inputs[var_idx]
            delta = delta_rel*np.abs(base_inputs[var_idx]) + 1e-4

            inputs_plus = list(base_inputs)
            inputs_plus[var_idx] = inputs_plus[var_idx] + delta
            rate_plus = calc_fn(*inputs_plus)

            inputs_minus = list(base_inputs)
            inputs_minus[var_idx] = inputs_minus[var_idx] - delta
            rate_minus = calc_fn(*inputs_minus)

            delta_broadcast = delta[:, None] if delta.ndim == 1 else delta
            d_log_rate = (rate_plus - rate_minus) / (2 * delta_broadcast)
            v_type = var_types.get(var_name, 'Gact')
            val_broadcast = base_val_array[:, None] if base_val_array.ndim == 1 else base_val_array

            if v_type == 'Gact':
                X_val = -(kb_eV * T) * d_log_rate
            elif v_type == 'resistance':
                X_val = -val_broadcast * d_log_rate
            else:
                X_val = val_broadcast * d_log_rate
                
            X_dict[var_name] = X_val

        rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2}
        with plt.rc_context(rc_update):
            fig, axes = plt.subplots(nrows=n_P, ncols=n_C, figsize=figsize, sharex='col', sharey=True)
            axes_2d = _axes_to_2d(axes, n_P, n_C)
            fig.suptitle('Degree of Rate Control', fontsize=title_size, fontweight='bold', y=0.97)
            fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size + 2)
            fig.supylabel(r'X$_{RC}$', fontweight='bold', fontsize=label_size + 2)

            colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
            index_map = {}
            _idx = 0
            for C in conc_list:
                for P in p_list:
                    L = len(self.state['truncated_E_exp'][(C, P)])
                    index_map[(C, P)] = (_idx, _idx + L)
                    _idx += L

            for i, C_val in enumerate(conc_list):
                for j, P_val in enumerate(p_list):
                    ax = axes_2d[j, i]
                    start, end = index_map[(C_val, P_val)]
                    E_slice = self.state['truncated_E_exp'][(C_val, P_val)]
                    sum_all = np.sum([X_dict[var][:, start:end] for var in perturb_vars], axis=0)
                    mu_sum = sum_all.mean(axis=0)
                    hdi_s = az.hdi(sum_all.reshape(n_chains, n_draws, -1), hdi_prob=0.95)
                    lower_s, upper_s = hdi_s[:, 0], hdi_s[:, 1]
                    ax.fill_between(E_slice, lower_s, upper_s, color='k', alpha=0.12, linewidth=0, zorder=6)
                    ax.plot(E_slice, mu_sum, color='gray', linestyle=':', lw=2, alpha=0.8)
                    for k, var in enumerate(perturb_vars):
                        col = colors[k % len(colors)]
                        data_slice = X_dict[var][:, start:end]
                        mu = data_slice.mean(axis=0)
                        hdi_rxn = az.hdi(data_slice.reshape(n_chains, n_draws, -1), hdi_prob=0.95)
                        lower, upper = hdi_rxn[:, 0], hdi_rxn[:, 1]
                        ax.fill_between(E_slice, lower, upper, color=col, alpha=0.15, linewidth=0)
                        ax.plot(E_slice, mu, color=col, lw=2.5, label=perturb_labels[k])
                    if j == 0:
                        ax.set_title(f'{C_val} {self.cfg["concentration_label"]}', fontsize=label_size, fontweight='bold')
                    if i == n_C - 1:
                        ax.text(1.05, 0.5, f'{P_val} atm', transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')
                    if i == 0 and j == 0:
                        ax.legend(loc='best', frameon=False, fontsize=10)
            plt.tight_layout(); ax.set_ylim([-0.05, 1.05])
            plt.subplots_adjust(right=0.92, top=0.90, bottom=0.08, left=0.10)
            plt.show()

def build_context(env_type):
    cfg = _resolve_config(env_type)
    return PlottingContext(cfg)
