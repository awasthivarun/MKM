# core.py
class CoreContext:
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