# plotting.py
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
from .config import _axes_to_2d

class PlottingMixin:

    def plot_model_fits(self, trace, ppc, loo, consolidated=False):
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
            if var_fit in ['log_rate', 'alpha', 'rate', 'log_residual'] or is_ppc:
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
            elif var_fit == 'loo':
                loo_vals = loo.loo_i.values
                if loo_vals.ndim > 1:
                    if loo_vals.shape[0] < loo_vals.shape[1]:
                        loo_vals = loo_vals.sum(axis=0) 
                    else:
                        loo_vals = loo_vals.sum(axis=1)
                idx = 0
                for C in conc_list:
                    for P in p_list:
                        length = len(self.state['truncated_E_exp'][(C, P)])
                        if C == C_val and P == P_val:
                            return loo_vals[idx:idx + length], self.state['truncated_E_exp'][(C, P)]
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
                    
                    if var_fit == 'loo':
                        loo_slice, E_model = get_model_slice(var_fit, C_val, P_val, is_ppc=False)
                        ax.scatter(E_model, loo_slice, color=colors[j], alpha=0.8, edgecolor='k', s=30, zorder=5)

                    else: 
                        model_slice, E_model = get_model_slice(var_fit, C_val, P_val, is_ppc=is_ppc)
                        model_mean = model_slice.mean(dim=("chain", "draw")) # Median is also good here!
                        hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                        hdi_90 = az.hdi(model_slice, hdi_prob=0.90)[model_slice.name]

                        if var_fit == 'log_residual':
                            ax.axhline(0, color='black', linestyle='--', lw=2, alpha=0.7, zorder=2)
                            ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[j], alpha=0.15, linewidth=0, zorder=3)
                            ax.fill_between(E_model, hdi_90[:, 0].values, hdi_90[:, 1].values, color=colors[j], alpha=0.30, linewidth=0, zorder=4)
                            ax.plot(E_model, model_mean, color=colors[j], lw=3.2, zorder=5)
                        
                        else: 
                            mean_key, sd_key, E_key = var_map[var_fit]
                            exp_mean = self.state[mean_key][cond_key]
                            exp_sd = self.state[sd_key][cond_key]
                            E_exp = self.state[E_key][cond_key]

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
            plot_grid('loo', n_P, n_C, figsize_main, "Pointwise ELPD", "Pointwise LOO-CV (ELPD)")
            plot_grid(ppc_var, n_P, n_C, figsize_main, "TOF (1/s)" if ppc_var == 'rate' else "log Rate", "Rate" if ppc_var == 'rate' else "Log Rate")
            plot_grid(non_ppc_var, n_P, n_C, figsize_main, "log Rate" if non_ppc_var == 'log_rate' else "TOF (1/s)", "Log Rate" if non_ppc_var == 'log_rate' else "Rate")
            plot_grid('log_residual', n_P, n_C, figsize_main, "Exp - Model", "Log Residual")
            if consolidated:
                plot_grid('alpha', 1, n_C, figsize_cons, 'alpha', 'Transfer Coefficients', consolidated=True)
                if n_C > 1: plot_grid(delta_name, 1, 1, figsize_1x1, self.cfg['delta_ylabel'], self.cfg['delta_title'], consolidated=True)
                if n_P > 1: plot_grid('delta_CO', 1, n_C, figsize_cons, 'Order (CO)', 'CO Reaction Order', consolidated=True)
            else:
                plot_grid('alpha', n_P, n_C, figsize_main, 'alpha', 'Transfer Coefficients')
                if n_C > 1: plot_grid(delta_name, 1, n_P, figsize_OH, self.cfg['delta_ylabel'], self.cfg['delta_title'])
                if n_P > 1: plot_grid('delta_CO', n_P - 1, n_C, figsize_CO, 'Order (CO)', 'CO Reaction Order')

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

            cov_vars = ['theta_CO', 'theta_OH', 'theta_COOH', 'theta_empty', 'theta_OH_star', 'theta_OH_pound', 'theta_O_star']
            cov_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:gray', 'tab:blue', 'tab:pink', 'tab:gray']
            cov_labels = [r'$\theta_{CO}$', r'$\theta_{OH}$', r'$\theta_{COOH}$', r'$\theta_{*}$', r'$\theta^*_{OH}$', r'$\theta^\#_{OH}$', r'$\theta^*_{O}$']

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

    