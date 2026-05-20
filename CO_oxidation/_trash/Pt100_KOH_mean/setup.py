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

plot_target = 'word'

# Experimental conditions and constants 
C_KOH_list = np.array([0.1, 0.25, 0.5, 1]) # in M 
P_CO_list = 0.01*np.array([0.1, 1, 10, 100]) # in atm
R, T, F = 8.3145, 293.15, 96485
kb_eV, h, kb_J = 8.617e-5, 6.626e-34, 1.3806e-23  # eV/K, J.s, J/K
N_A = 6.022e23  # Avogadro's number

# Load and parse data
with open('import_Pt100_KOH_replicates.pkl', 'rb') as f:
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
E_OH_exp = {}
delta_OH_exp = {}
delta_OH_SD_exp = {}
E_CO_exp = {}
delta_CO_exp = {}
delta_CO_SD_exp = {}

for C in C_KOH_list:
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
        E_OH_exp[cond] = data['E_OH']
        delta_OH_exp[cond] = data['delta_OH']
        delta_OH_SD_exp[cond] = data['delta_OH_SD']
        if 'E_CO' in data:
            E_CO_exp[cond] = data['E_CO']
            delta_CO_exp[cond] = data['delta_CO']
            delta_CO_SD_exp[cond] = data['delta_CO_SD']

E_in = np.concatenate([truncated_E_exp[(C, P)] for C in C_KOH_list for P in P_CO_list])
C_KOH_in = np.concatenate([np.full_like(truncated_E_exp[(C, P)], C) for C in C_KOH_list for P in P_CO_list])
P_CO_in = np.concatenate([np.full_like(truncated_E_exp[(C, P)], P) for C in C_KOH_list for P in P_CO_list])
rate_obs = np.concatenate([truncated_rate_exp[(C, P)] for C in C_KOH_list for P in P_CO_list])
rate_SD_obs = np.concatenate([truncated_rate_SD_exp[(C, P)] for C in C_KOH_list for P in P_CO_list])
log_rate_obs = np.concatenate([truncated_log_rate_exp[(C, P)] for C in C_KOH_list for P in P_CO_list])
log_rate_SD_obs = np.concatenate([truncated_log_rate_SD_exp[(C, P)] for C in C_KOH_list for P in P_CO_list])

rate_obs_matrix = np.concatenate([truncated_rate_matrix[(C, P)] for C in C_KOH_list for P in P_CO_list], axis=1)

def add_post_sampling_observables(trace):
    """Calculate deterministic observables after sampling."""

    # 1. Calculate alpha
    log_rate_samples = trace.posterior['log_rate'].values
    alpha_samples = np.zeros_like(log_rate_samples)
    model_log_rates = {}
    start = 0
    for C in C_KOH_list:
        for P in P_CO_list:
            E_arr = truncated_E_exp[(C, P)]
            end = start + len(E_arr)
            alpha_samples[:, :, start:end] = (R * T / F) * np.gradient(log_rate_samples[:, :, start:end], E_arr[1] - E_arr[0], axis=2)
            model_log_rates[(C, P)] = log_rate_samples[:, :, start:end]
            start = end
    trace.posterior['alpha'] = (trace.posterior['log_rate'].dims, alpha_samples)
    
    # 2. Calculate delta_OH
    delta_OH_list = []
    ln_C = np.log(C_KOH_list)
    w_C = ((ln_C - np.mean(ln_C)) / np.sum((ln_C - np.mean(ln_C))**2))[None, None, :, None]
    for P in P_CO_list:
        E_arrays = [np.round(truncated_E_exp[(C, P)], 6) for C in C_KOH_list]
        common_E = E_arrays[0]
        for arr in E_arrays[1:]: common_E = np.intersect1d(common_E, arr)   
        rates_for_fit = []
        for C in C_KOH_list:
            E_arr = np.round(truncated_E_exp[(C, P)], 6)
            mask = np.isin(E_arr, common_E)
            rates_for_fit.append(model_log_rates[(C, P)][:, :, mask])    
        rates_for_fit = np.stack(rates_for_fit, axis=2)
        delta_OH = np.sum(w_C * rates_for_fit, axis=2)
        delta_OH_list.append(delta_OH)
    trace.posterior['delta_OH'] = (("chain", "draw", "delta_OH_flat"), np.concatenate(delta_OH_list, axis=2))
    
    # 3. Calculate delta_CO
    delta_CO_list = []
    for C in C_KOH_list:
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

def plot_posteriors(trace, model, target='word'): 

    all_vars = list(trace.posterior.data_vars)
    excluded_vars = ['alpha', 'delta_OH', 'delta_CO', 'rate', 'log_rate']
    var_names = [v for v in all_vars if not v.endswith("__")]
    kinetic_vars = [v for v in var_names if not v.startswith(('theta','phi')) and v not in excluded_vars]
    summary = az.summary(trace, var_names=kinetic_vars)
    print(summary)
    
    num_vars = len(kinetic_vars)
    cols = min(4, max(1, num_vars)) 
    rows = int(np.ceil(num_vars / cols)) 
    if target == 'ppt':
        rc_update = {'font.size': 14, 'axes.linewidth': 1.5, 'lines.linewidth': 2}
        figsize_post = (3.0 * cols, 3.5 * rows) 
        title_size = 20; text_size_adj = 12 
    else:
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

def plot_model_fits(trace, ppc, target='word', consolidated=False):

    if target == 'ppt':
        plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5, 'lines.linewidth': 2.5})
        figsize_4x4 = (12, 11); figsize_3x4 = (12, 8.5); figsize_1x4 = (12, 4)   
        title_size, label_size = 18, 14
    else:
        plt.rcParams.update({'font.size': 10, 'axes.linewidth': 1, 'lines.linewidth': 2})
        figsize_4x4 = (7.5, 7); figsize_3x4 = (7.5, 5.5); figsize_1x4 = (7.5, 2.5)
        title_size, label_size = 12, 10
    colors = plt.cm.tab10.colors[:len(P_CO_list)]
    ci_multiplier = 3.182 / np.sqrt(4)
    ppc_vars = [v for v in ppc.observed_data.data_vars if not v.endswith("__")]
    if len(ppc_vars) != 1:
        raise ValueError(f'Expected exactly one PPC observed variable, found {ppc_vars}')
    ppc_var = ppc_vars[0]
    non_ppc_var = 'log_rate' if ppc_var == 'rate' else 'rate'

    def get_model_slice(var_fit, C_KOH, P_CO, is_ppc=False):
        dataset = ppc.posterior_predictive if is_ppc else trace.posterior
        track_name = ppc_var if is_ppc else var_fit
        if var_fit in ['log_rate', 'alpha', 'rate'] or is_ppc:
            
            data_arr = dataset[track_name]
            if is_ppc and data_arr.ndim == 4:
                if data_arr.shape[2] == 4: 
                    data_arr = data_arr.isel(**{data_arr.dims[2]: 0})
                elif data_arr.shape[3] == 4: 
                    data_arr = data_arr.isel(**{data_arr.dims[3]: 0})
            idx = 0
            for C in C_KOH_list:
                for P in P_CO_list:
                    length = len(truncated_E_exp[(C, P)])
                    if C == C_KOH and P == P_CO:
                        return data_arr[:, :, idx:idx+length], truncated_E_exp[(C, P)]
                    idx += length
        elif var_fit == 'delta_OH':
            idx = 0
            for P in P_CO_list:
                E_arrays = [np.round(truncated_E_exp[(C, P)], 6) for C in C_KOH_list]
                common_E = E_arrays[0]
                for arr in E_arrays[1:]: common_E = np.intersect1d(common_E, arr)
                length = len(common_E)
                if P == P_CO: return dataset[var_fit][:, :, idx:idx+length], common_E
                idx += length
        elif var_fit == 'delta_CO':
            idx = 0
            for C in C_KOH_list:
                for i in range(len(P_CO_list)-1):
                    P1, P2 = P_CO_list[i], P_CO_list[i+1]
                    E1 = np.round(truncated_E_exp[(C, P1)], 6)
                    E2 = np.round(truncated_E_exp[(C, P2)], 6)
                    common_E = np.intersect1d(E1, E2)
                    length = len(common_E)
                    if C == C_KOH and P1 == P_CO: return dataset[var_fit][:, :, idx:idx+length], common_E
                    idx += length

    def plot_grid(var_fit, nrows, ncols, figsize, ylabel, title_text, is_3x4=False, is_1x4=False, consolidated=False):
        is_ppc = (var_fit == ppc_var)
        share_y = False if is_ppc else True
        linestyles = ['-', '--', ':']  # For consolidated delta_CO: solid, dashed, dotted (or dots)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex='col', sharey=share_y)
        fig.suptitle(title_text, fontsize=title_size, fontweight='bold', y=0.96) # Adjusted y down slightly
        fig.supxlabel(r"Potential (V$_{\mathbf{SHE}}$)", fontweight='bold', fontsize=label_size + 2)
        fig.supylabel(ylabel, fontweight='bold', fontsize=label_size + 2)

        if consolidated and var_fit == 'delta_CO':
            for i, C_KOH in enumerate(C_KOH_list):
                ax = axes[i] if nrows == 1 else axes[0, i]
                for pressure_idx in range(len(P_CO_list) - 1):
                    P1, P2 = P_CO_list[pressure_idx], P_CO_list[pressure_idx + 1]
                    cond_key = (C_KOH, P1)
                    exp_mean = delta_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_KOH, P1, is_ppc=is_ppc)
                    model_mean = model_slice.mean(dim=("chain", "draw"))
                    hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                    ax.plot(E_model, model_mean, color=colors[pressure_idx], lw=3.2, ls=linestyles[pressure_idx], zorder=5)
                    ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[pressure_idx], alpha=0.15, linewidth=0, zorder=3)
                    ax.plot(E_model, exp_mean, color='black', lw=2.5, ls=linestyles[pressure_idx], alpha=0.6, zorder=2)
                if i == 0: ax.set_title(f'{C_KOH} M KOH', fontsize=label_size, fontweight='bold')
            plt.tight_layout(); plt.subplots_adjust(right=0.92, top=0.90, left=0.10, bottom=0.08); plt.show()
            return

        for i, C_KOH in enumerate(C_KOH_list):
            for j, P_CO in enumerate(P_CO_list):
                if is_1x4 and i > 0: continue
                if is_3x4 and j >= 3: continue

                if is_1x4:
                    ax = axes[j]
                    cond_key = (C_KOH_list[1], P_CO) 
                    exp_mean = delta_OH_exp[cond_key]
                    exp_sd = delta_OH_SD_exp[cond_key]
                    E_exp = E_OH_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, None, P_CO, is_ppc=is_ppc)
                elif is_3x4:
                    ax = axes[j, i]
                    cond_key = (C_KOH, P_CO)
                    exp_mean = delta_CO_exp[cond_key]
                    exp_sd = delta_CO_SD_exp[cond_key]
                    E_exp = E_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_KOH, P_CO, is_ppc=is_ppc)
                else:
                    ax = axes[j, i]
                    cond_key = (C_KOH, P_CO)
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
                    elif var_fit == 'delta_OH':
                        exp_mean = delta_OH_exp[cond_key]
                        exp_sd = delta_OH_SD_exp[cond_key]
                        E_exp = E_OH_exp[cond_key]
                    elif var_fit == 'delta_CO':
                        exp_mean = delta_CO_exp[cond_key]
                        exp_sd = delta_CO_SD_exp[cond_key]
                        E_exp = E_CO_exp[cond_key]
                    model_slice, E_model = get_model_slice(var_fit, C_KOH, P_CO, is_ppc=is_ppc)

                model_mean = model_slice.mean(dim=("chain", "draw"))
                hdi_95 = az.hdi(model_slice, hdi_prob=0.95)[model_slice.name]
                hdi_90 = az.hdi(model_slice, hdi_prob=0.90)[model_slice.name]

                # if var_fit != ppc_var and not consolidated:
                if not consolidated:  
                    exp_ci = exp_sd * ci_multiplier
                    ax.fill_between(E_exp, exp_mean - exp_ci, exp_mean + exp_ci, color='gray', alpha=0.3, linewidth=0, zorder=1)
                # Plot the experimental mean line for everything
                ax.plot(E_exp, exp_mean, color='black', lw=2.5, alpha=0.6, zorder=2)

                # 2. Model Data Handling (HDI shading is plotted over the truncated Nernstian bounds)
                ax.fill_between(E_model, hdi_95[:, 0].values, hdi_95[:, 1].values, color=colors[j], alpha=0.15, linewidth=0, zorder=3)
                ax.fill_between(E_model, hdi_90[:, 0].values, hdi_90[:, 1].values, color=colors[j], alpha=0.30, linewidth=0, zorder=4)
                ax.plot(E_model, model_mean, color=colors[j], lw=3.2, zorder=5)
                if j == 0 and not is_1x4: ax.set_title(f'{C_KOH} M KOH', fontsize=label_size, fontweight='bold')
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
    plot_grid('alpha', 4, 4, figsize_4x4, "alpha", "Transfer Coefficients")
    plot_grid('delta_OH', 1, 4, figsize_1x4, "Order (OH)", "OH Reaction Order", is_1x4=True)
    if consolidated:
        plot_grid('delta_CO', 1, 4, figsize_3x4, "Order (CO)", "CO Reaction Order", consolidated=True)
    else:
        plot_grid('delta_CO', 3, 4, figsize_3x4, "Order (CO)", "CO Reaction Order", is_3x4=True)
    plt.rcParams.update(plt.rcParamsDefault)

def plot_coverages(trace, target='word'):
    
    if target == 'ppt':
        plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5, 'lines.linewidth': 2.5})
        figsize = (12, 11)
        title_size, label_size = 20, 16
    else:
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
    for C in C_KOH_list:
        for P in P_CO_list:
            L = len(truncated_E_exp[(C, P)])
            index_map[(C, P)] = (_idx, _idx + L)
            _idx += L

    cov_vars = ['theta_CO', 'theta_OH', 'theta_COOH', 'theta_empty']
    cov_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:gray']
    cov_labels = [r'$\theta_{CO}$', r'$\theta_{OH}$', r'$\theta_{COOH}$', r'$\theta_{*}$']

    def _hdi_from_vals(vals_2d, prob=0.95):
        # vals_2d: (samples, points)
        try:
            da = xr.DataArray(vals_2d, dims=("draw", "point"))
            hdi = az.hdi(da, hdi_prob=prob)
            hdi = np.asarray(hdi)
            if hdi.ndim == 2 and hdi.shape[1] == 2:
                return hdi[:, 0], hdi[:, 1]
            if hdi.ndim == 1 and hdi.shape[0] == 2:
                return hdi[0], hdi[1]
        except Exception:
            pass
        pct = np.percentile(vals_2d, [100 * (1 - prob) / 2, 100 * (1 + prob) / 2], axis=0)
        return pct[0], pct[1]

    for i, C_KOH in enumerate(C_KOH_list):
        for j, P_CO in enumerate(P_CO_list):
            ax = axes[j, i]
            start, end = index_map[(C_KOH, P_CO)]

            sum_vals = None
            for v, col, lab in zip(cov_vars, cov_colors, cov_labels):
                if v not in trace.posterior:
                    continue
                data = trace.posterior[v][:, :, start:end]
                mu = data.mean(dim=("chain", "draw"))
                vals = data.values
                flat = vals.reshape(-1, vals.shape[-1])
                lower, upper = _hdi_from_vals(flat, prob=0.95)
                ax.fill_between(truncated_E_exp[(C_KOH, P_CO)], lower, upper, color=col, alpha=0.15, linewidth=0)
                ax.plot(truncated_E_exp[(C_KOH, P_CO)], mu, color=col, label=lab, lw=2.5)

                sum_vals = vals.copy() if sum_vals is None else sum_vals + vals

            if sum_vals is not None:
                mu_sum = np.mean(sum_vals, axis=(0, 1))
                flat_sum = sum_vals.reshape(-1, sum_vals.shape[-1])
                lower_s, upper_s = _hdi_from_vals(flat_sum, prob=0.95)
                ax.fill_between(truncated_E_exp[(C_KOH, P_CO)], lower_s, upper_s, color='k', alpha=0.12, linewidth=0, zorder=6)
                ax.plot(truncated_E_exp[(C_KOH, P_CO)], mu_sum, color='k', lw=1, ls='--', zorder=7)

            ax.set_ylim(-0.05, 1.05)
            if j == 0:
                ax.set_title(f'{C_KOH} M KOH', fontsize=label_size, fontweight='bold')
            if i == 3:
                ax.text(1.05, 0.5, f'{P_CO} atm', transform=ax.transAxes, rotation=-90, va='center', fontsize=label_size, fontweight='bold')
            if i == 0 and j == 0:
                ax.legend(loc='best', frameon=False, fontsize=10)

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
    rate = pt.exp(log_rate)
    sigma = pm.Gamma('sigma', alpha=2, beta=20)
    # sigma_total = pt.sqrt(rate_SD_obs**2 + (rate*sigma)**2)
    rate = pm.StudentT('rate', nu=4, mu=rate, sigma=sigma*rate, observed=rate_obs)
    return rate