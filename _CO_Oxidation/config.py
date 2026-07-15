# config.py
import numpy as np
import pandas as pd 

R, T, F = 8.3145, 293.15, 96485
kb_eV, h, kb_J = 8.617e-5, 6.626e-34, 1.3806e-23
N_A = 6.022e23

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

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