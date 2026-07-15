# drc.py
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pytensor
from pytensor.graph import vectorize_graph
from .config import _axes_to_2d, kb_eV, T

class DRCMixin:
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