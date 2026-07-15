# processing.py
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
from .config import R, T, F

pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

class ProcessingMixin:

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
        exp_log_rates_list = []
        for C in conc_list:
            for P in p_list:
                exp_log_rates_list.append(self.state['truncated_log_rate_exp'][(C, P)])
        
        exp_log_rates_flat = np.concatenate(exp_log_rates_list)
        log_residuals = exp_log_rates_flat[None, None, :] - log_rate_samples
        trace.posterior['log_residual'] = (trace.posterior['log_rate'].dims, log_residuals)

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
        
    def simulate_fake_data(self, model):
        print("Simulating fake data from prior means...")
        with model:
            prior = pm.sample_prior_predictive(samples=5000)
        
        fake_trace_dict = {}
        print("--- True Parameters (Prior Mu) ---")
        free_rv_names = [rv.name for rv in model.free_RVs]
        for var in prior.prior.data_vars:
            if str(var) in free_rv_names:
                if str(var) == 'sigma_rel': val = 0.01
                else: val = prior.prior[var].mean().values
                fake_trace_dict[str(var)] = np.array([[val]])
                print(f"{var}: {val:.3f}")
      
        fake_trace = az.from_dict(posterior=fake_trace_dict)
        with model:
            ppc = pm.sample_posterior_predictive(fake_trace)    
        rate_var = [v for v in ppc.posterior_predictive.data_vars if not str(v).endswith("__")][0]
        new_obs = ppc.posterior_predictive[rate_var].values[0, 0]
        
        # Overwrite the global state so the next model instantiation uses the fake data
        self.state['rate_obs_matrix'] = new_obs
        print("IMPORTANT: Re-run 'with pm.Model() as ModelName:' cell to bake the fake data into the likelihood, then run fit_and_evaluate().")