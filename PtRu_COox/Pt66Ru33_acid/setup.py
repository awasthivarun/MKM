import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import pickle
import nutpie
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import pytensor
import xarray as xr

pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); pd.set_option('display.max_colwidth', None)
pd.options.mode.string_storage = "python"; pd.options.future.infer_string = False

# Experimental conditions and constants 
C_H_list = np.array([0.1, 0.25, 0.5, 1]) # in M 
P_CO_list = 0.01*np.array([0.1, 1, 10, 100]) # in atm
R, T, F = 8.3145, 293.15, 96485
kb_eV, h, kb_J = 8.617e-5, 6.626e-34, 1.3806e-23  # eV/K, J.s, J/K
N_A = 6.022e23  # Avogadro's number

# Load and parse data
with open('import_Pt66Ru33_HClO4.pkl', 'rb') as f:
    experiments = pickle.load(f)

truncated_E_exp = {}
truncated_rate_exp = {}
truncated_rate_SD_exp = {}
truncated_log_rate_exp = {}
truncated_log_rate_SD_exp = {}
truncated_rate_matrix = {}
E_exp_map = {}
rate_exp = {}
rate_SD_exp = {}
log_rate_exp = {}
log_rate_SD_exp = {}
alpha_exp = {}
alpha_SD_exp = {}
E_H_exp = {}
delta_H_exp = {}
delta_H_SD_exp = {}
E_CO_exp = {}
delta_CO_exp = {}
delta_CO_SD_exp = {}

for C in C_H_list:
    for P in P_CO_list:
        cond = (C, P)
        data = experiments[cond]
        truncated_E_exp[cond] = data['truncated_E']
        truncated_rate_exp[cond] = data['truncated_rate']
        truncated_rate_SD_exp[cond] = data['truncated_rate_SD']
        truncated_log_rate_exp[cond] = data['truncated_log_rate']
        truncated_log_rate_SD_exp[cond] = data['truncated_log_rate_SD']
        truncated_rate_matrix[cond] = data['truncated_rate_matrix']
        E_exp_map[cond] = data['E']
        rate_exp[cond] = data['rate']
        rate_SD_exp[cond] = data['rate_SD']
        log_rate_exp[cond] = data['log_rate']
        log_rate_SD_exp[cond] = data['log_rate_SD']
        alpha_exp[cond] = data['alpha']
        alpha_SD_exp[cond] = data['alpha_SD']
        E_H_exp[cond] = data['E_H']
        delta_H_exp[cond] = data['delta_H']
        delta_H_SD_exp[cond] = data['delta_H_SD']
        if 'E_CO' in data:
            E_CO_exp[cond] = data['E_CO']
            delta_CO_exp[cond] = data['delta_CO']
            delta_CO_SD_exp[cond] = data['delta_CO_SD']

E_in = np.concatenate([truncated_E_exp[(C, P)] for C in C_H_list for P in P_CO_list])
C_H_in = np.concatenate([np.full_like(truncated_E_exp[(C, P)], C) for C in C_H_list for P in P_CO_list])
P_CO_in = np.concatenate([np.full_like(truncated_E_exp[(C, P)], P) for C in C_H_list for P in P_CO_list])
rate_obs = np.concatenate([truncated_rate_exp[(C, P)] for C in C_H_list for P in P_CO_list])
rate_SD_obs = np.concatenate([truncated_rate_SD_exp[(C, P)] for C in C_H_list for P in P_CO_list])
log_rate_obs = np.concatenate([truncated_log_rate_exp[(C, P)] for C in C_H_list for P in P_CO_list])
log_rate_SD_obs = np.concatenate([truncated_log_rate_SD_exp[(C, P)] for C in C_H_list for P in P_CO_list])

rate_obs_matrix = np.concatenate([truncated_rate_matrix[(C, P)] for C in C_H_list for P in P_CO_list], axis=1)

def add_post_sampling_observables(trace):
    """Calculate deterministic observables after sampling."""

    # 1. Calculate alpha
    log_rate_samples = trace.posterior['log_rate'].values
    alpha_samples = np.zeros_like(log_rate_samples)
    model_log_rates = {}
    start = 0
    for C in C_H_list:
        for P in P_CO_list:
            E_arr = truncated_E_exp[(C, P)]
            end = start + len(E_arr)
            alpha_samples[:, :, start:end] = (R * T / F) * np.gradient(log_rate_samples[:, :, start:end], E_arr, axis=2)
            model_log_rates[(C, P)] = log_rate_samples[:, :, start:end]
            start = end
    trace.posterior['alpha'] = (trace.posterior['log_rate'].dims, alpha_samples)
    
    # 2. Calculate delta_H
    delta_H_list = []
    ln_C = np.log(C_H_list)
    w_C = ((ln_C - np.mean(ln_C)) / np.sum((ln_C - np.mean(ln_C))**2))[None, None, :, None]
    for P in P_CO_list:
        E_arrays = [np.round(truncated_E_exp[(C, P)], 6) for C in C_H_list]
        common_E = E_arrays[0]
        for arr in E_arrays[1:]: common_E = np.intersect1d(common_E, arr)   
        rates_for_fit = []
        for C in C_H_list:
            E_arr = np.round(truncated_E_exp[(C, P)], 6)
            mask = np.isin(E_arr, common_E)
            rates_for_fit.append(model_log_rates[(C, P)][:, :, mask])    
        rates_for_fit = np.stack(rates_for_fit, axis=2)
        delta_H = np.sum(w_C * rates_for_fit, axis=2)
        delta_H_list.append(delta_H)
    trace.posterior['delta_H'] = (("chain", "draw", "delta_H_flat"), np.concatenate(delta_H_list, axis=2))
    
    # 3. Calculate delta_CO
    delta_CO_list = []
    for C in C_H_list:
        for i in range(len(P_CO_list) - 1):
            P1, P2 = P_CO_list[i], P_CO_list[i+1]
            E1 = np.round(truncated_E_exp[(C, P1)], 6)
            E2 = np.round(truncated_E_exp[(C, P2)], 6)
            common_E = np.intersect1d(E1, E2)
            mask1 = np.isin(E1, common_E); mask2 = np.isin(E2, common_E)
            lr1 = model_log_rates[(C, P1)][:, :, mask1]
            lr2 = model_log_rates[(C, P2)][:, :, mask2]
            delta_CO = (lr2 - lr1) / np.log(P2 / P1)
            delta_CO_list.append(delta_CO)       
    trace.posterior['delta_CO'] = (("chain", "draw", "delta_CO_flat"), np.concatenate(delta_CO_list, axis=2))
    
    return trace

def calculate_flattened_r2(ppc, var_name):
    y_true = ppc.observed_data[var_name].values
    y_pred = ppc.posterior_predictive[var_name].values
    chains, draws = y_pred.shape[:2]
    y_pred_flat = y_pred.reshape(chains * draws, -1)
    y_true_flat = y_true.flatten()
    return az.r2_score(y_true_flat, y_pred_flat)['r2']

def _hdi_from_vals(vals_2d, prob=0.95):
    try:
        da = xr.DataArray(vals_2d, dims=("draw", "point"))
        hdi = np.asarray(az.hdi(da, hdi_prob=prob))
        if hdi.ndim == 2 and hdi.shape[1] == 2:
            return hdi[:, 0], hdi[:, 1]
        if hdi.ndim == 1 and hdi.shape[0] == 2:
            return hdi[0], hdi[1]
    except Exception:
        pass
    pct = np.percentile(vals_2d, [100 * (1 - prob) / 2, 100 * (1 + prob) / 2], axis=0)
    return pct[0], pct[1]

def plot_posteriors(trace, model): 

    all_vars = list(trace.posterior.data_vars)
    excluded_vars = ['alpha', 'delta_H', 'delta_CO', 'rate', 'log_rate']
    var_names = [v for v in all_vars if not v.endswith("__")]
    kinetic_vars = [v for v in var_names if not v.startswith(('theta','phi')) and v not in excluded_vars]
    summary = az.summary(trace, var_names=kinetic_vars)
    print(summary)
    
    num_vars = len(kinetic_vars)
    cols = min(4, max(1, num_vars)) 
    rows = int(np.ceil(num_vars / cols)) 
    rc_update = {'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 1.5}
    figsize_post = (2.5 * cols, 3.0 * rows) 
    title_size = 14; text_size_adj = 10

    # Use a context manager so we don't permanently alter or break notebook plotting behavior
    with plt.rc_context(rc_update):
        if kinetic_vars:
            axes = az.plot_posterior(trace, var_names=kinetic_vars, hdi_prob=0.95, round_to=3, figsize=figsize_post, grid=(rows, cols), textsize=text_size_adj)
            axes_flat = np.array(axes).flatten() 
            for ax in axes_flat: 
                if ax is not None and hasattr(ax, 'texts'):
                    for text_obj in ax.texts: 
                        if 'mean' in text_obj.get_text(): text_obj.set_fontweight('bold')
                    ax.title.set_fontweight('bold')     
            plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.88); plt.show()
        if len(kinetic_vars) > 1:
            plot_pair_vars = [v for v in kinetic_vars if v not in ['CO_converge_error', 'OH_converge_error']]
            az.plot_pair(trace, var_names=plot_pair_vars, kind='kde', divergences=True, figsize=(8,8), textsize=10); plt.show()

    with model: ppc = pm.sample_posterior_predictive(trace, progressbar=False)
    for var_name in ppc.observed_data.data_vars:
        r2 = calculate_flattened_r2(ppc, var_name)
        print(f"{var_name}: {r2:.3f}")
      
    return ppc

def plot_model_fits(trace, ppc, consolidated=False):

    plt.rcParams.update({'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2})
    figsize_4x4 = (7.5, 7); figsize_3x4 = (7.5, 5.5); 
    figsize_1x4 = (7.5, 2.5); figsize_2x2 = (7.5/1.3, 7/1.3); 
    figsize_1x1 = (4, 3.6)  
    title_size, label_size = 12, 10

    colors = plt.cm.tab10.colors[:len(P_CO_list)]
    ci_multiplier = 1 # Just using +- SD for experimental data
    ppc_vars = [v for v in ppc.observed_data.data_vars if not v.endswith("__")]
    if len(ppc_vars) != 1:
        raise ValueError(f'Expected exactly one PPC observed variable, found {ppc_vars}')
    ppc_var = ppc_vars[0]
    non_ppc_var = 'log_rate' if ppc_var == 'rate' else 'rate'

    def get_model_slice(var_fit, C_H, P_CO, is_ppc=False):
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
            for C in C_H_list:
                for P in P_CO_list:
                    length = len(truncated_E_exp[(C, P)])
                    if C == C_H and P == P_CO:
                        return data_arr[:, :, idx:idx+length], truncated_E_exp[(C, P)]
                    idx += length
        
        elif var_fit == 'delta_H':
            idx = 0
            for P in P_CO_list:
                E_arrays = [np.round(truncated_E_exp[(C, P)], 6) for C in C_H_list]
                common_E = E_arrays[0]
                for arr in E_arrays[1:]: common_E = np.intersect1d(common_E, arr)
                length = len(common_E)
                if P == P_CO: return dataset[var_fit][:, :, idx:idx+length], common_E
                idx += length
        
        elif var_fit == 'delta_CO':
            idx = 0
            for C in C_H_list:
                for i in range(len(P_CO_list)-1):
                    P1, P2 = P_CO_list[i], P_CO_list[i+1]
                    E1 = np.round(truncated_E_exp[(C, P1)], 6)
                    E2 = np.round(truncated_E_exp[(C, P2)], 6)
                    common_E = np.intersect1d(E1, E2)
                    length = len(common_E)
                    if C == C_H and P1 == P_CO: return dataset[var_fit][:, :, idx:idx+length], common_E
                    idx += length

    def plot_grid(var_fit, nrows, ncols, figsize, ylabel, title_text, is_3x4=False, is_1x4=False, consolidated=False):
        is_ppc = (var_fit == ppc_var)
        share_y = False if is_ppc else True
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex='col', sharey=share_y)
        fig.suptitle(title_text, fontsize=title_size, fontweight='bold', y=0.96)
        fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size + 2)
        fig.supylabel(ylabel, fontweight='bold', fontsize=label_size + 2)

        if consolidated and var_fit == 'alpha':
            legend_handles = []
            linestyles = [':', (0, (1, 0.5)), '--', '-']
            for i, C_H in enumerate(C_H_list):
                row, col = i // 2, i % 2
                ax = axes[row, col]

                for pressure_idx in range(len(P_CO_list)):
                    P_val = P_CO_list[pressure_idx]
                    cond_key = (C_H, P_val)
                    exp_mean = alpha_exp[cond_key]
                    E_exp = E_exp_map[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_H, P_val, is_ppc=is_ppc)
                    model_mean = model_slice.mean(dim=("chain", "draw"))
                    hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]

                    line, = ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5, label=f'{P_val} atm')
                    ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                    ax.plot(E_exp, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.20, zorder=2)
                    if i == 0: legend_handles.append(line)

                ax.text(0.95, 0.95, f'{C_H} M Acid', transform=ax.transAxes, fontsize=label_size, fontweight='bold', va='top', ha='right')
                
                ax.set_yticks([-0.5, 0, 0.5])
                for label in ax.get_yticklabels(): label.set_fontweight('bold')
                for label in ax.get_xticklabels(): label.set_fontweight('bold')
                if i == 3: ax.legend(handles=legend_handles, loc='lower left', frameon=False, prop={'size': label_size, 'weight': 'bold'})

            plt.tight_layout()
            plt.subplots_adjust(right=0.92, top=0.91, left=0.12, bottom=0.1)
            plt.show()
            return

        if consolidated and var_fit == 'delta_CO':
            legend_handles = []
            linestyles = [':', '--', '-']
            for i, C_H in enumerate(C_H_list):
                row, col = i // 2, i % 2
                ax = axes[row, col]

                for pressure_idx in range(len(P_CO_list) - 1):
                    P1, P2 = P_CO_list[pressure_idx], P_CO_list[pressure_idx + 1]
                    cond_key = (C_H, P1)
                    exp_mean = delta_CO_exp[cond_key]
                    E_exp = E_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_H, P1, is_ppc=is_ppc)
                    model_mean = model_slice.mean(dim=("chain", "draw"))
                    hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]

                    line, = ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5, label=f'{P1} - {P2} atm')
                    ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                    ax.plot(E_exp, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.20, zorder=2)
                    if i == 0: legend_handles.append(line)

                ax.text(0.05, 0.95, f'{C_H} M Acid', transform=ax.transAxes, fontsize=label_size, fontweight='bold', va='top', ha='left')
                ticks = [0.0, 0.5, 1.0]
                ax.set_yticks(ticks)
                ax.set_yticklabels(ticks, fontweight='bold')
                for label in ax.get_xticklabels(): label.set_fontweight('bold')

            plt.tight_layout()
            plt.subplots_adjust(right=0.92, top=0.91, left=0.12, bottom=0.14)
            fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=len(legend_handles), frameon=False, prop={'size': label_size, 'weight': 'bold'})
            plt.show()
            return
        
        if consolidated and var_fit == 'delta_H':
            legend_handles = []
            linestyles = [':', (0, (1, 0.5)), '--', '-']

            C_H = C_H_list[0] # Any random C_H
            row, col = 1, 1
            ax = axes
            for pressure_idx in range(len(P_CO_list)):
                P_val = P_CO_list[pressure_idx]
                cond_key = (C_H, P_val)
                exp_mean = delta_H_exp[cond_key]
                E_exp = E_H_exp[cond_key]
                model_slice, E_model = get_model_slice(var_fit, C_H, P_val, is_ppc=is_ppc)
                model_mean = model_slice.mean(dim=("chain", "draw"))
                hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]

                line, = ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5, label=f'{P_val} atm')
                ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                ax.plot(E_exp, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.20, zorder=2)
                legend_handles.append(line)
                
            ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, fontweight='bold')
            ticks = [-0.2, -0.1, 0.0, 0.1]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.92, top=0.9, left=0.15, bottom=0.14)
            fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.12), ncol=2, frameon=False, prop={'size': label_size, 'weight': 'bold'})
            plt.show()
            return

        for i, C_H in enumerate(C_H_list):
            for j, P_CO in enumerate(P_CO_list):
                if is_1x4 and i > 0: continue
                if is_3x4 and j >= 3: continue

                if is_1x4:
                    ax = axes[j]
                    cond_key = (C_H_list[1], P_CO) 
                    exp_mean = delta_H_exp[cond_key]
                    exp_sd = delta_H_SD_exp[cond_key]
                    E_exp = E_H_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, None, P_CO, is_ppc=is_ppc)
                elif is_3x4:
                    ax = axes[j, i]
                    cond_key = (C_H, P_CO)
                    exp_mean = delta_CO_exp[cond_key]
                    exp_sd = delta_CO_SD_exp[cond_key]
                    E_exp = E_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_H, P_CO, is_ppc=is_ppc)
                else:
                    ax = axes[j, i]
                    cond_key = (C_H, P_CO)
                    if var_fit == 'log_rate':
                        exp_mean = log_rate_exp[cond_key]
                        exp_sd = log_rate_SD_exp[cond_key]
                        E_exp = E_exp_map[cond_key]
                    elif var_fit == 'alpha':
                        exp_mean = alpha_exp[cond_key]
                        exp_sd = alpha_SD_exp[cond_key]
                        E_exp = E_exp_map[cond_key]
                    elif var_fit == 'rate':
                        exp_mean = rate_exp[cond_key]
                        exp_sd = rate_SD_exp[cond_key]
                        E_exp = E_exp_map[cond_key]
                    elif var_fit == 'delta_H':
                        exp_mean = delta_H_exp[cond_key]
                        exp_sd = delta_H_SD_exp[cond_key]
                        E_exp = E_H_exp[cond_key]
                    elif var_fit == 'delta_CO':
                        exp_mean = delta_CO_exp[cond_key]
                        exp_sd = delta_CO_SD_exp[cond_key]
                        E_exp = E_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_H, P_CO, is_ppc=is_ppc)

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
                
                if j == 0 and not is_1x4: ax.set_title(f'{C_H} M Acid', fontsize=label_size, fontweight='bold')
                if is_1x4: ax.set_title(f'{P_CO} atm', fontsize=label_size, fontweight='bold')
                if i == 3 and not is_1x4:
                    label_right = f'{P_CO} atm' if not is_3x4 else f'{P_CO_list[j]} - {P_CO_list[j+1]} atm'
                    ax.text(1.05, 0.5, label_right, transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')

        bottom_adj = 0.08
        if is_1x4: top_adj = 0.80; bottom_adj = 0.2
        elif is_3x4: top_adj = 0.88; bottom_adj = 0.10
        else: top_adj = 0.90
        plt.tight_layout(); plt.subplots_adjust(right=0.92, top=top_adj, left=0.10, bottom=bottom_adj); plt.show()

    plot_grid(ppc_var, 4, 4, figsize_4x4, "TOF (1/s)" if ppc_var == 'rate' else "log Rate", "Rate" if ppc_var == 'rate' else "Log Rate")
    plot_grid(non_ppc_var, 4, 4, figsize_4x4, "log Rate" if non_ppc_var == 'log_rate' else "TOF (1/s)", "Log Rate" if non_ppc_var == 'log_rate' else "Rate")
    if consolidated:
        plot_grid('alpha', 2, 2, figsize_2x2, "alpha", "Transfer Coefficients", consolidated=True)
        plot_grid('delta_H', 1, 1, figsize_1x1, "Order (H+)", "Proton Reaction Order", consolidated=True)
        plot_grid('delta_CO', 2, 2, figsize_2x2, "Order (CO)", "CO Reaction Order", consolidated=True)
    else:
        plot_grid('alpha', 4, 4, figsize_4x4, "alpha", "Transfer Coefficients")
        plot_grid('delta_H', 1, 4, figsize_1x4, "Order (H+)", "Proton Reaction Order", is_1x4=True)
        plot_grid('delta_CO', 3, 4, figsize_3x4, "Order (CO)", "CO Reaction Order", is_3x4=True)
    plt.rcParams.update(plt.rcParamsDefault)

def plot_coverages(trace):
    
    plt.rcParams.update({'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2})
    figsize = (7.5, 6.5)
    title_size, label_size = 12, 10

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=figsize, sharex='col', sharey=True)
    fig.suptitle('Modeled Surface Coverages', fontsize=title_size, fontweight='bold', y=0.97)
    fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size+2)
    fig.supylabel("Coverage", fontweight='bold', fontsize=label_size+2)

    # build index map for slicing posterior arrays
    index_map = {}
    _idx = 0
    for C in C_H_list:
        for P in P_CO_list:
            L = len(truncated_E_exp[(C, P)])
            index_map[(C, P)] = (_idx, _idx + L)
            _idx += L

    cov_vars = ['theta_CO', 'theta_COOH', 'theta_OH_Pt', 'theta_OH_Ru']
    cov_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:cyan']
    cov_labels = [r'$\theta_{CO}$', r'$\theta_{COOH}$', r'$\theta_{OH,Pt}$', r'$\theta_{OH,Ru}$']

    for i, C_H in enumerate(C_H_list):
        for j, P_CO in enumerate(P_CO_list):
            ax = axes[j, i]
            start, end = index_map[(C_H, P_CO)]

            sum_vals = None
            for v, col, lab in zip(cov_vars, cov_colors, cov_labels):
                if v not in trace.posterior:
                    continue
                data = trace.posterior[v][:, :, start:end]
                mu = data.mean(dim=("chain", "draw"))
                vals = data.values
                flat = vals.reshape(-1, vals.shape[-1])
                lower, upper = _hdi_from_vals(flat, prob=0.95)
                ax.fill_between(truncated_E_exp[(C_H, P_CO)], lower, upper, color=col, alpha=0.15, linewidth=0)
                ax.plot(truncated_E_exp[(C_H, P_CO)], mu, color=col, label=lab, lw=2.5)

                sum_vals = vals.copy() if sum_vals is None else sum_vals + vals

            if sum_vals is not None:
                mu_sum = np.mean(sum_vals, axis=(0, 1))
                flat_sum = sum_vals.reshape(-1, sum_vals.shape[-1])
                lower_s, upper_s = _hdi_from_vals(flat_sum, prob=0.95)
                ax.fill_between(truncated_E_exp[(C_H, P_CO)], lower_s, upper_s, color='k', alpha=0.12, linewidth=0, zorder=6)
                ax.plot(truncated_E_exp[(C_H, P_CO)], mu_sum, color='k', lw=1, ls='--', zorder=7)

            ax.set_ylim(-0.05, 1.05)
            if j == 0:
                ax.set_title(f'{C_H} M HClO$_4$', fontsize=label_size, fontweight='bold')
            if i == 3:
                ax.text(1.05, 0.5, f'{P_CO} atm', transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')
            if i == 0 and j == 0:
                ax.legend(loc='best', frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.92, top=0.90, bottom=0.08, left=0.10)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def plot_drc(model, trace, perturb_vars, perturb_labels=None, var_types=None):
    """
    Parameters:
    - var_types: dict mapping variable names to mathematical relationships. 
                 'Gact' applies -kb*T * d_ln_r/d_Gact.
                 'resistance' applies -theta * d_ln_r/d_theta (e.g., boundary layer).
                 'promoter' applies theta * d_ln_r/d_theta (e.g., pre-exponential).
                 Defaults to 'Gact' for all perturb_vars.
                 Example: var_types={'Gact2_LH_0': 'Gact', 'delta_BL_0': 'resistance'}
    """
    plt.rcParams.update({'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2})
    figsize = (7.5, 6.5)
    title_size, label_size = 12, 10

    if var_types is None:
        var_types = {v: 'Gact' for v in perturb_vars}
    if perturb_labels is None:
        perturb_labels = [v.replace('_0', '') for v in perturb_vars]

    delta = 1e-4
    target_node = model['log_rate']
    replacements = {}
    sym_inputs = []
    input_names = []

    for rv in model.free_RVs:
        if rv.name in trace.posterior:
            sym_var = rv.type() 
            sym_inputs.append(sym_var)
            replacements[rv] = sym_var
            input_names.append(rv.name)
            
            if rv.name in model.named_vars:
                replacements[model[rv.name]] = sym_var

    for var_name in perturb_vars:
        if var_name in input_names:
            idx = input_names.index(var_name)
            replacements[model[var_name]] = sym_inputs[idx]

    cloned_node = pytensor.clone_replace(target_node, replace=replacements)
    calc_fn = pytensor.function(sym_inputs, cloned_node, on_unused_input='ignore')

    post = trace.posterior
    n_chains, n_draws = post.sizes["chain"], post.sizes["draw"]
    total_samples = n_chains * n_draws

    trace_data = {name: post[name].values.reshape(total_samples, *post[name].shape[2:]) for name in input_names}
    X_dict = {v: [] for v in perturb_vars}

    for idx in range(total_samples):
        base_vals = [trace_data[name][idx] for name in input_names]
        
        for i, var_name in enumerate(perturb_vars):
            var_idx = input_names.index(var_name)
            base_val = base_vals[var_idx]
            
            vals_plus = list(base_vals)
            vals_plus[var_idx] += delta
            rate_plus = calc_fn(*vals_plus)
            
            vals_minus = list(base_vals)
            vals_minus[var_idx] -= delta
            rate_minus = calc_fn(*vals_minus)
            
            d_log_rate = (rate_plus - rate_minus) / (2 * delta)
            v_type = var_types.get(var_name, 'Gact')
            
            if v_type == 'Gact':
                X_val = - (kb_eV * T) * d_log_rate
            elif v_type == 'resistance': 
                X_val = - base_val * d_log_rate
            elif v_type == 'promoter':
                X_val = base_val * d_log_rate
                
            X_dict[var_name].append(X_val)

    for v in perturb_vars:
        X_dict[v] = np.array(X_dict[v])

    fig, axes = plt.subplots(nrows=len(P_CO_list), ncols=len(C_H_list), figsize=figsize, sharex='col', sharey=True)
    fig.suptitle('Degree of Rate Control', fontsize=title_size, fontweight='bold', y=0.97)
    fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size+2)
    fig.supylabel(r"X$_{RC}$", fontweight='bold', fontsize=label_size+2)

    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    index_map = {}
    _idx = 0
    
    for C in C_H_list:
        for P in P_CO_list:
            L = len(truncated_E_exp[(C, P)])
            index_map[(C, P)] = (_idx, _idx + L)
            _idx += L

    for i, C_H in enumerate(C_H_list):
        for j, P_CO in enumerate(P_CO_list):
            ax = axes[j, i]
            start, end = index_map[(C_H, P_CO)]
            E_slice = truncated_E_exp[(C_H, P_CO)]

            # compute sum across perturb_vars for each posterior sample, then plot mean and 95% HDI
            sum_all = np.sum([X_dict[var][:, start:end] for var in perturb_vars], axis=0)  # shape: (n_samples, n_points)
            mu_sum = sum_all.mean(axis=0)
            flat_sum = sum_all.reshape(-1, sum_all.shape[-1])
            lower_s, upper_s = _hdi_from_vals(flat_sum, prob=0.95)
            ax.fill_between(E_slice, lower_s, upper_s, color='k', alpha=0.12, linewidth=0, zorder=6)
            ax.plot(E_slice, mu_sum, color='gray', linestyle=':', lw=2, alpha=0.8)

            for k, var in enumerate(perturb_vars):
                col = colors[k % len(colors)]
                data_slice = X_dict[var][:, start:end]
                mu = data_slice.mean(axis=0)

                lower, upper = _hdi_from_vals(data_slice, prob=0.95)
                ax.fill_between(E_slice, lower, upper, color=col, alpha=0.15, linewidth=0)
                ax.plot(E_slice, mu, color=col, lw=2.5, label=perturb_labels[k])
            
            if j == 0: ax.set_title(f'{C_H} M HClO$_4$', fontsize=label_size, fontweight='bold')
            if i == 3: ax.text(1.05, 0.5, f'{P_CO} atm', transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')
            if i == 0 and j == 0: ax.legend(loc='best', frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.92, top=0.90, bottom=0.08, left=0.10)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def fit_and_evaluate(model, draws=1000, tune=2000, chains=4, cores=4, init_mean=None, target_accept=0.9):
    compiled_model = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled_model, draws=draws, tune=tune, chains=chains, cores=cores, init_mean=init_mean, target_accept=target_accept)
    pm.compute_log_likelihood(trace, progressbar=False, model=model)
    loo = az.loo(trace, pointwise=True)
    print(loo); az.plot_khat(loo)
    plt.title("Pareto k diagnostic (Global Model)"); plt.show()
    trace = add_post_sampling_observables(trace)
    return trace, loo

# All models are comparable as long as this function remains the same
def observables(log_rate):
    pm.Deterministic('log_rate', log_rate)
    rate_model = pt.exp(log_rate)

    sigma_rel = pm.Gamma('sigma_rel', alpha=2, beta=20)
    sigma_base = pm.HalfNormal('sigma_base', sigma=1e-2)
    exponent =  pm.HalfNormal('exponent', sigma=1)
    sigma_model = pt.sqrt(sigma_base**2 + (sigma_rel**2)*(rate_model ** exponent))
    # sigma_total = pt.sqrt(rate_SD_obs**2 + sigma_model**2)
    
    rate = pm.StudentT('rate', nu=2, mu=rate_model, sigma=sigma_model, observed=rate_obs_matrix)
    return rate