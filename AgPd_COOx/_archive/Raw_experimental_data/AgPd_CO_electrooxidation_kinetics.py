#!/usr/bin/env python
# coding: utf-8
"""
AgPd CO electro-oxidation kinetics -- full analysis pipeline (script version).

Plain-Python equivalent of AgPd_CO_electrooxidation_kinetics.ipynb, for IDEs
without Jupyter (VS Code, PyCharm, Spyder, ...) or a plain terminal.

Run it from inside the folder that holds this file, Experiment_Key.xlsx and
the <KOH> data folders:

    python AgPd_CO_electrooxidation_kinetics.py

It reads the raw .txt data, runs the pipeline, and writes every publication
figure as a .png into that same folder. Runs on Windows, macOS, and Linux.
"""

# In[1]:


# Imports
import glob
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import linregress, theilslopes
from matplotlib.lines import Line2D

# Publication plot styling (matches templates/)
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.weight']      = 'bold'
mpl.rcParams['figure.dpi']       = 1200

# Shared CO concentration metadata (ordered: 100%, 10%, 1%, 0.1%)
co_labels    = ['100', '10', '1', '0pt1']
co_display   = ['100%', '10%', '1%', '0.1%']
co_filenames = ['CO 100%', 'CO 10%', 'CO 1%', 'CO 0.1%']

# Catalyst order, display names, and colors — matches CV_stats_250.ipynb template
catalysts  = ['Pd100', 'Ag10Pd90', 'Ag25Pd75', 'Ag50Pd50', 'Ag75Pd25', 'Ag90Pd10']
cat_names  = [r'Pd$_{100}$', r'Ag$_{10}$Pd$_{90}$', r'Ag$_{25}$Pd$_{75}$',
              r'Ag$_{50}$Pd$_{50}$', r'Ag$_{75}$Pd$_{25}$', r'Ag$_{90}$Pd$_{10}$']
cat_colors = [(0,0,0), (0.4,0.0,0.4), (0.7,0.0,0.7),
              (0.0,0.0,0.6), (0.0,0.0,1.0), (0.0,0.5,1.0)]

# Pure-Ag (Ag100) is ingested but excluded from main-text/SI plots; see
# dedicated Ag100 cell at the end of the notebook.
ag_catalysts  = ['Ag100']
ag_cat_names  = [r'Ag$_{100}$']
ag_cat_colors = [('g')]

# Experiment Key + data root — resolved from this file's own folder.
# Keep this file inside the folder holding Experiment_Key.xlsx and the
# <KOH> data folders. __file__ is defined when run as a script; if this
# code is pasted into a Jupyter cell where __file__ is undefined, fall
# back to cwd (Jupyter sets cwd to the notebook's folder when opened).
try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_DIR = os.getcwd()
EXPERIMENT_KEY_PATH = os.path.join(PROJECT_DIR, 'Experiment_Key.xlsx')

# Common voltage grid used by all recast DataFrames (V vs RHE, IR-corrected)
commonRHE = np.linspace(0.50, 0.99, 1000)
# Common voltage grid for OH-dependence analysis (V vs SHE)
commonSHE = np.linspace(-0.30, 0.20, 1000)

# Physical constants (transfer coefficient calc — matches processing template)
F = 96485.3321; R = 8.314; T = 20 + 273.15


# In[2]:


def read_cv(path, file, ref_pot):
    """Read an EC-Lab text-export CV file and return a dict keyed by cycle.

    Detects the header length from the 'Nb header lines' field at the top
    of the file. Returns ``{'Cycle Number N': ndarray(:, 2)}`` where each
    array has columns (Ewe vs RHE, I).
    """
    filepath = os.path.join(path, file)
    nb_header = None
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith('Nb header lines'):
                nb_header = int(line.split(':')[1].strip())
                break
    if nb_header is None:
        raise ValueError(f'"Nb header lines" not found in {file}')

    df = pd.read_csv(filepath, sep='\t', skiprows=nb_header - 1,
                     encoding='latin-1', low_memory=False)

    cycle_dict = {}
    cycles = sorted(df['cycle number'].dropna().unique())

    if cycles == [0.0] or not cycles:
        cv = np.zeros((len(df), 2))
        cv[:, 0] = df['Ewe/V'].values + ref_pot
        cv[:, 1] = df['<I>/mA'].values
        cycle_dict['Cycle Number 1'] = cv
    else:
        for i, cyc in enumerate(cycles):
            sub = df[df['cycle number'] == cyc]
            cv = np.zeros((len(sub), 2))
            cv[:, 0] = sub['Ewe/V'].values + ref_pot
            cv[:, 1] = sub['<I>/mA'].values
            cycle_dict['Cycle Number ' + str(i + 1)] = cv

    return cycle_dict


# In[3]:


def process_run(run_dir, comp=80, threshold=0.25):
    """Run end-to-end processing for a single experiment folder.

    1. Parse path  -> directory / catalyst / run letter
    2. Look up RHE calibration, Ru, SHE shift in the Experiment Key
    3. Read CO 100/10/1/0.1% and stripping .txt files
    4. Compute ECSA from CO stripping integration
    5. Apply IR correction; build processed DataFrames
    6. Recast onto ``commonRHE``; build recast DataFrames

    Returns
    -------
    dict with keys:
        ECSA, Ru, SHE_shift : float
        processed : {label: DataFrame}  — IR-corrected full cycle
        recast    : {label: DataFrame}  — interpolated onto commonRHE (anodic only)
    """
    parts        = os.path.normpath(run_dir).split(os.sep)
    directory    = parts[-3]
    subdirectory = parts[-2]
    run_letter   = parts[-1].replace('Run', '')

    Experiment_Key = pd.read_excel(EXPERIMENT_KEY_PATH, sheet_name='Key')
    key_row = Experiment_Key[
        (Experiment_Key['Directory']    == directory)    &
        (Experiment_Key['Subdirectory'] == subdirectory) &
        (Experiment_Key['Run']          == run_letter)
    ]
    if key_row.empty:
        raise ValueError(f'No Experiment Key entry for {directory} / {subdirectory} / Run{run_letter}')

    Hg_vs_RHE = float(key_row['RHE Calibration (Ref v RHE)'].values[0])
    Ru        = float(key_row['Ru'].values[0])
    SHE_shift = float(key_row['SHE Shift'].values[0])

    cycle   = 'Cycle Number 1'
    path1   = run_dir
    ref_pot = Hg_vs_RHE

    COdf100 = pd.DataFrame(read_cv(path1, 'CO 100%.txt', ref_pot)[cycle], columns=['Ewe_RHE', 'I'])
    COdf10  = pd.DataFrame(read_cv(path1, 'CO 10%.txt',  ref_pot)[cycle], columns=['Ewe_RHE', 'I'])
    COdf1   = pd.DataFrame(read_cv(path1, 'CO 1%.txt',   ref_pot)[cycle], columns=['Ewe_RHE', 'I'])
    COdfpt1 = pd.DataFrame(read_cv(path1, 'CO 0.1%.txt', ref_pot)[cycle], columns=['Ewe_RHE', 'I'])

    strip_data = read_cv(path1, 'CO stripping.txt', ref_pot)
    stripdf    = pd.DataFrame(strip_data[cycle], columns=['Ewe_RHE', 'I'])
    if 'Cycle Number 2' in strip_data:
        strip2 = pd.DataFrame(strip_data['Cycle Number 2'], columns=['Ewe_RHE', 'I'])
        n = min(len(stripdf), len(strip2))
        stripdf = stripdf.iloc[:n].copy()
        stripdf['Ewe_RHE_2'] = strip2['Ewe_RHE'].values[:n]
        stripdf['I_2']       = strip2['I'].values[:n]

    # ECSA from CO stripping
    stripdfa = pd.DataFrame()
    stripdfa['Ewe_RHE_2'] = stripdf['Ewe_RHE_2'].loc[0:stripdf['Ewe_RHE_2'].idxmax()]
    stripdfa['I_2']       = stripdf['I_2'].loc[0:stripdf['Ewe_RHE_2'].idxmax()]
    Vt = stripdfa[(stripdfa['Ewe_RHE_2'] <= threshold) &
                  (stripdfa['Ewe_RHE_2'] <= stripdfa['Ewe_RHE_2'].idxmax())].index
    stripdfa.drop(Vt, inplace=True)
    stripdfa.reset_index(inplace=True)

    stripdfb = pd.DataFrame()
    stripdfb['Ewe_RHE'] = stripdf['Ewe_RHE'].loc[0:stripdf['Ewe_RHE'].idxmax()]
    stripdfb['I']       = stripdf['I'].loc[0:stripdf['Ewe_RHE'].idxmax()]
    Vtb = stripdfb[(stripdfb['Ewe_RHE'] <= threshold) &
                   (stripdfb['Ewe_RHE'] <= stripdfb['Ewe_RHE'].idxmax())].index
    stripdfb.drop(Vtb, inplace=True)
    stripdfb.reset_index(inplace=True)

    stripdfc = pd.DataFrame()
    stripdfc['Ewe_RHE']    = stripdfb['Ewe_RHE']
    stripdfc['I']          = stripdfb['I']
    stripdfc['Ewe_RHE_2']  = stripdfa['Ewe_RHE_2']
    stripdfc['I_2']        = stripdfa['I_2']
    stripdfc['difference'] = stripdfc['I'] - stripdfc['I_2']

    Vthreshdiff = stripdfc[stripdfc['difference'] <= 0].index
    Area = pd.DataFrame()
    Area['Ewe']      = stripdfc['Ewe_RHE'].drop(Vthreshdiff)
    Area['pos_diff'] = stripdfc['difference'].drop(Vthreshdiff)
    Area.reset_index(inplace=True)
    Area = Area.drop('index', axis=1)
    Area['Running integral'] = integrate.cumulative_trapezoid(Area.pos_diff, Area.Ewe, initial=0)
    Area['C']    = Area['Running integral'] * (1000/20) * 1000
    Area['ECSA'] = Area['C'] / 420
    ECSA = Area['ECSA'].max()

    co_dfs = [COdf100, COdf10, COdf1, COdfpt1]
    processed = {}
    recast    = {}

    # --- Blank CV (both cycles, IR-corrected, ECSA-normalized) ---
    blank_data = read_cv(path1, 'Blank.txt', ref_pot)
    blank1 = pd.DataFrame(blank_data[cycle], columns=['Ewe_RHE', 'I'])
    blank_df = pd.DataFrame()
    blank_df['Ewe_RHE']      = blank1['Ewe_RHE']
    blank_df['I']            = blank1['I']
    blank_df['ImuA']         = blank1['I'] * 1000
    blank_df['Ewe_RHE_corr'] = blank1['Ewe_RHE'] - np.absolute(blank1['I'])/1000 * Ru * (100-comp)/100
    blank_df['Ewe_SHE_corr'] = blank_df['Ewe_RHE_corr'] - SHE_shift
    blank_df['j1uA']         = blank1['I'] / ECSA * 1000
    if 'Cycle Number 2' in blank_data:
        blank2 = pd.DataFrame(blank_data['Cycle Number 2'], columns=['Ewe_RHE', 'I'])
        # Keep cycle 1 at full length; NaN-pad cycle 2 columns to match
        n1 = len(blank_df)
        ewe2 = np.full(n1, np.nan)
        i2   = np.full(n1, np.nan)
        nfit = min(n1, len(blank2))
        ewe2[:nfit] = blank2['Ewe_RHE'].values[:nfit]
        i2[:nfit]   = blank2['I'].values[:nfit]
        blank_df['Ewe_RHE_2']     = ewe2
        blank_df['I_2']           = i2
        blank_df['Ewe_RHE_corr2'] = ewe2 - np.absolute(i2)/1000 * Ru * (100-comp)/100
        blank_df['j2uA']          = i2 / ECSA * 1000
    processed['blank'] = blank_df.copy()

    # --- CO stripping CV (both cycles, IR-corrected, ECSA-normalized) ---
    strip_plot = pd.DataFrame()
    strip_plot['Ewe_RHE_1'] = stripdf['Ewe_RHE']
    strip_plot['I_1']       = stripdf['I']
    strip_plot['Ewe1c']     = stripdf['Ewe_RHE'] - np.absolute(stripdf['I'])/1000 * Ru * (100-comp)/100
    strip_plot['j1uA']      = stripdf['I'] / ECSA * 1000
    if 'Ewe_RHE_2' in stripdf.columns:
        strip_plot['Ewe_RHE_2'] = stripdf['Ewe_RHE_2']
        strip_plot['I_2']       = stripdf['I_2']
        strip_plot['Ewe2c']     = stripdf['Ewe_RHE_2'] - np.absolute(stripdf['I_2'])/1000 * Ru * (100-comp)/100
        strip_plot['j2uA']      = stripdf['I_2'] / ECSA * 1000
    processed['stripping'] = strip_plot


    for label, df in zip(co_labels, co_dfs):
        # IR correction
        df['ImuA']         = df['I'] * 1000
        df['Ewe_RHE_corr'] = df['Ewe_RHE'] - np.absolute(df['I'])/1000 * Ru * (100-comp)/100
        df['Ewe_SHE_corr'] = df['Ewe_RHE_corr'] - SHE_shift
        df['juA']          = df['I'] / ECSA * 1000
        df['Logj']         = np.log(np.absolute(df['juA']))
        df['alpha']        = (df['Logj'].diff().bfill() /
                              df['Ewe_RHE_corr'].diff().bfill()) * (R * T / F)
        processed[label]   = df[['Ewe_RHE', 'Ewe_RHE_corr', 'Ewe_SHE_corr',
                                  'I', 'ImuA', 'juA', 'Logj', 'alpha']].copy()

        # Recast onto commonRHE (anodic sweep only)
        idx      = df.Ewe_RHE_corr.idxmax()
        x        = df.Ewe_RHE_corr.loc[0:idx].values
        y        = df.juA.loc[0:idx].values
        f_interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
        fit      = pd.DataFrame({
            'Ewe_RHE_corr': commonRHE,
            'Ewe_SHE_corr': commonRHE - SHE_shift,
            f'j_{label}':   f_interp(commonRHE),
        })
        fit['Logj']   = np.log(np.absolute(fit[f'j_{label}']))
        recast[label] = fit

    print(f'  Run{run_letter}: ECSA = {ECSA:.4f} cm^2, Ru = {Ru:.4f} ohm, SHE_shift = {SHE_shift:.4f} V')
    return {'ECSA': ECSA, 'Ru': Ru, 'SHE_shift': SHE_shift,
            'processed': processed, 'recast': recast}


# In[4]:


def build_avg_combined(run_results):
    """Average recast DataFrames across runs for each CO concentration.

    Parameters
    ----------
    run_results : list of dicts
        Each dict is the return value of ``process_run``.

    Returns
    -------
    DataFrame
        Columns: Ewe_RHE_corr, Ewe_SHE_corr,
                 j_mean_<label>, j_std_<label>  for each CO label.
        All rows are on the shared ``commonRHE`` voltage grid.
    """
    avg = pd.DataFrame()
    avg['Ewe_RHE_corr'] = run_results[0]['recast']['100']['Ewe_RHE_corr'].values
    avg['Ewe_SHE_corr'] = run_results[0]['recast']['100']['Ewe_SHE_corr'].values
    for label in co_labels:
        j_stack = np.column_stack(
            [r['recast'][label][f'j_{label}'].values for r in run_results]
        )
        avg[f'j_mean_{label}'] = np.nanmean(j_stack, axis=1)
        avg[f'j_std_{label}']  = np.nanstd(j_stack,  axis=1)
    return avg


# In[5]:


def build_avg_processed(run_results):
    """Average processed DataFrames across runs onto commonRHE grid (no fit/recast).

    At each grid point, the mean/std are computed only if ALL runs have data
    there — otherwise NaN. Prevents erratic averages near the top of the
    voltage range where a subset of runs reach further than others.
    """
    rows = {'Ewe_RHE_corr': commonRHE}
    for label in co_labels:
        stacks = []
        for rr in run_results:
            df  = rr['processed'][label]
            idx = df['Ewe_RHE_corr'].idxmax()
            xx  = df['Ewe_RHE_corr'].loc[:idx].values
            yy  = df['juA'].loc[:idx].values
            fi  = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
            stacks.append(fi(commonRHE))
        mat = np.array(stacks)
        all_valid = ~np.isnan(mat).any(axis=0)
        mean_arr = np.full(mat.shape[1], np.nan)
        std_arr  = np.full(mat.shape[1], np.nan)
        if all_valid.any():
            mean_arr[all_valid] = mat[:, all_valid].mean(axis=0)
            std_arr[all_valid]  = mat[:, all_valid].std(axis=0)
        rows[f'j_mean_{label}'] = mean_arr
        rows[f'j_std_{label}']  = std_arr
    return pd.DataFrame(rows)


# In[6]:


def build_avg_alpha(run_results):
    """Average per-run alpha onto commonRHE grid. Mean/std at each grid point
    require ALL runs to have valid (non-NaN) data — same rule as build_avg_processed.
    alpha definition matches the processing template: (d ln|j|/dE) * RT/F.
    """
    rows = {'Ewe_RHE_corr': commonRHE}
    for label in co_labels:
        stacks = []
        for rr in run_results:
            df  = rr['processed'][label]
            idx = df['Ewe_RHE_corr'].idxmax()
            xx  = df['Ewe_RHE_corr'].loc[:idx].values
            yy  = df['alpha'].loc[:idx].values
            fi  = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
            stacks.append(fi(commonRHE))
        mat = np.array(stacks)
        all_valid = ~np.isnan(mat).any(axis=0)
        mean_arr = np.full(mat.shape[1], np.nan)
        std_arr  = np.full(mat.shape[1], np.nan)
        if all_valid.any():
            mean_arr[all_valid] = mat[:, all_valid].mean(axis=0)
            std_arr[all_valid]  = mat[:, all_valid].std(axis=0)
        rows[f'alpha_mean_{label}'] = mean_arr
        rows[f'alpha_std_{label}']  = std_arr
    return pd.DataFrame(rows)


# In[7]:


def build_co_order_steps(run_results):
    """CO reaction order (per adjacent step pair) from triplicate per-run data.

    For each pair (low, high) in [(0.1%, 1%), (1%, 10%), (10%, 100%)]:
        At each commonRHE potential, gather 6 points (3 runs × 2 CO% levels) and
        linregress log|j| (µA/cm²) vs log(P_CO in Pa). Store slope and stderr.

    All-runs-valid rule: if any of the 6 points is NaN, slope and stderr are NaN.

    Returns DataFrame on commonRHE with columns:
        Ewe_RHE_corr,
        CO_Order_{lo}_{hi}, CO_Order_{lo}_{hi}_err  for each pair.
    """
    P_total = 101325.0  # Pa (1 atm)
    co_frac = {'100': 1.0, '10': 0.1, '1': 0.01, '0pt1': 0.001}
    P_Pa    = {lbl: frac * P_total for lbl, frac in co_frac.items()}

    # Per-run Logj interpolated onto commonRHE for each CO%
    stacks = {lbl: [] for lbl in co_frac}
    for rr in run_results:
        for lbl in co_frac:
            df  = rr['processed'][lbl]
            idx = df['Ewe_RHE_corr'].idxmax()
            xx  = df['Ewe_RHE_corr'].loc[:idx].values
            yy  = df['Logj'].loc[:idx].values
            fi  = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
            stacks[lbl].append(fi(commonRHE))
    stacks = {k: np.array(v) for k, v in stacks.items()}

    pairs = [('0pt1', '1'), ('1', '10'), ('10', '100')]
    n_grid = len(commonRHE)
    rows = {'Ewe_RHE_corr': commonRHE}
    for lo, hi in pairs:
        n_runs = stacks[lo].shape[0]
        x_vec  = np.concatenate([np.full(n_runs, np.log(P_Pa[lo])),
                                 np.full(n_runs, np.log(P_Pa[hi]))])
        slope_arr  = np.full(n_grid, np.nan)
        stderr_arr = np.full(n_grid, np.nan)
        for j in range(n_grid):
            y_vec = np.concatenate([stacks[lo][:, j], stacks[hi][:, j]])
            if not np.isnan(y_vec).any():
                lr = linregress(x_vec, y_vec)
                slope_arr[j]  = lr.slope
                stderr_arr[j] = lr.stderr
        rows[f'CO_Order_{lo}_{hi}']     = slope_arr
        rows[f'CO_Order_{lo}_{hi}_err'] = stderr_arr
    return pd.DataFrame(rows)


# In[8]:


def build_oh_order_global(koh_to_runs):
    """Global OH reaction order on the SHE potential grid.

    For each CO%:
      At each commonSHE point, regress log|j_uA| vs log([OH-] in mol/L) over
      all 3 KOH levels × 3 runs = 9 points. Stores slope and stderr.
    All-runs-valid rule: if any of the 9 points is NaN at a grid point,
    slope and stderr there are NaN.

    Parameters
    ----------
    koh_to_runs : dict
        {1000: [run_results...], 500: [run_results...], 250: [run_results...]}

    Returns
    -------
    DataFrame on commonSHE with columns:
        Ewe_SHE_corr,
        OH_Order_{co_label}, OH_Order_{co_label}_err  for each CO%.
    """
    KOH_M = {1000: 1.0, 500: 0.5, 250: 0.25}
    n_grid = len(commonSHE)

    rows = {'Ewe_SHE_corr': commonSHE}
    for co_label in co_labels:
        stack_y, x_list = [], []
        for koh, M in KOH_M.items():
            for rr in koh_to_runs[koh]:
                df  = rr['processed'][co_label]
                idx = df['Ewe_SHE_corr'].idxmax()
                xx  = df['Ewe_SHE_corr'].loc[:idx].values
                yy  = df['Logj'].loc[:idx].values
                fi  = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
                stack_y.append(fi(commonSHE))
                x_list.append(np.log(M))
        stack_y = np.array(stack_y)      # (9, n_grid)
        x_vec   = np.array(x_list)       # length 9

        slope_arr  = np.full(n_grid, np.nan)
        stderr_arr = np.full(n_grid, np.nan)
        for j in range(n_grid):
            y_vec = stack_y[:, j]
            if not np.isnan(y_vec).any():
                lr = linregress(x_vec, y_vec)
                slope_arr[j]  = lr.slope
                stderr_arr[j] = lr.stderr
        rows[f'OH_Order_{co_label}']     = slope_arr
        rows[f'OH_Order_{co_label}_err'] = stderr_arr
    return pd.DataFrame(rows)


# In[9]:


def build_avg_logj(run_results):
    """Average per-run Logj onto commonRHE grid (all-runs-valid masking)."""
    rows = {'Ewe_RHE_corr': commonRHE}
    for label in co_labels:
        stacks = []
        for rr in run_results:
            df  = rr['processed'][label]
            idx = df['Ewe_RHE_corr'].idxmax()
            xx  = df['Ewe_RHE_corr'].loc[:idx].values
            yy  = df['Logj'].loc[:idx].values
            fi  = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
            stacks.append(fi(commonRHE))
        mat = np.array(stacks)
        all_valid = ~np.isnan(mat).any(axis=0)
        mean_arr = np.full(mat.shape[1], np.nan)
        std_arr  = np.full(mat.shape[1], np.nan)
        if all_valid.any():
            mean_arr[all_valid] = mat[:, all_valid].mean(axis=0)
            std_arr[all_valid]  = mat[:, all_valid].std(axis=0)
        rows[f'Logj_mean_{label}'] = mean_arr
        rows[f'Logj_std_{label}']  = std_arr
    return pd.DataFrame(rows)



# In[10]:


# --- Figure decoration helpers ---
# Every plot cell calls decorate_figure(fig, title[, handles=...]) right before
# plt.savefig. Suptitle is drawn on every figure; pass `handles=...` to also
# draw a single-row legend above the axes. When a legend is requested the
# helper auto-leaves room at the top via subplots_adjust(top=...) so the
# legend body does not collide with the per-axis column titles.

def _legend_handles_alpha_line(items, linewidth=1):
    """Solid-line proxies with varying alpha. items: [(label, alpha), ...]."""
    return [Line2D([0], [0], color="black", alpha=a, linestyle="solid",
                   linewidth=linewidth, label=lbl) for lbl, a in items]

def _legend_handles_linestyle(items, linewidth=1):
    """Line proxies with varying linestyle. items: [(label, linestyle), ...]."""
    return [Line2D([0], [0], color="black", linestyle=ls,
                   linewidth=linewidth, label=lbl) for lbl, ls in items]

def _legend_handles_alpha_marker(items, marker="o", markersize=4):
    """Marker-only proxies with varying alpha. items: [(label, alpha), ...]."""
    return [Line2D([0], [0], color="black", marker=marker, linestyle="none",
                   markersize=markersize, markeredgecolor="black",
                   markeredgewidth=0.3, alpha=a, label=lbl) for lbl, a in items]

def decorate_figure(fig, title, *, handles=None, fontsize=10,
                    bbox_y=0.95, suptitle_y=0.99, ncols=None,
                    make_room=True, top=0.82):
    """Stamp a suptitle on the figure; optionally add an above-axes legend."""
    if handles is not None:
        if make_room:
            fig.subplots_adjust(top=top)
        if ncols is None:
            ncols = len(handles)
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, bbox_y), ncols=ncols, frameon=False,
                   fontsize=fontsize, handletextpad=0.4, columnspacing=1.2)
    fig.suptitle(title, y=suptitle_y, fontsize=fontsize, fontweight="bold")


# In[11]:


# --- Ingest all catalysts and all KOH concentrations ---
#
# all_run_data[koh_mM][catalyst] -> list of run-result dicts from process_run()
# all_data[koh_mM][catalyst]     -> avg_df from build_avg_combined()
#   columns: Ewe_RHE_corr, Ewe_SHE_corr, j_mean_<label>, j_std_<label>

batch_root = PROJECT_DIR  # data root = this script's folder (see top)

all_run_data = {}
all_data     = {}

for koh_dir in sorted(glob.glob(os.path.join(batch_root, '*'))):
    if not os.path.isdir(koh_dir):
        continue
    koh_name   = os.path.basename(koh_dir)
    conc_match = re.search(r'(\d{3,4})\s*mM', koh_name)
    if not conc_match:
        continue
    koh_mM = int(conc_match.group(1))
    all_run_data[koh_mM] = {}
    all_data[koh_mM]     = {}
    print(f'\n=== {koh_name} ===')

    for catalyst in catalysts + ag_catalysts:
        cat_dir  = os.path.join(koh_dir, catalyst)
        run_dirs = sorted(glob.glob(os.path.join(cat_dir, 'Run*')))
        if not run_dirs:
            print(f'  {catalyst}: no Run* directories — skipping')
            continue
        print(f'  {catalyst}:')
        run_results = [process_run(rd) for rd in run_dirs]
        all_run_data[koh_mM][catalyst] = run_results
        all_data[koh_mM][catalyst]     = build_avg_combined(run_results)

print(f'\nDone.')
for koh in sorted(all_data.keys(), reverse=True):
    print(f'  {koh} mM: {list(all_data[koh].keys())}')

# Convenience views for the Pd100-specific plot cells below
catalyst = 'Pd100'
koh_data = {koh: all_data[koh]['Pd100'] for koh in all_data}
run_data  = {koh: all_run_data[koh]['Pd100'] for koh in all_run_data}


# In[12]:


# --- Blank CV figure — voltage-based slicing of cycle 1 H-UPD region + full cycle 2.
# Layout: 6 rows (catalysts: Pd100 top → Ag90Pd10 bottom) × 3 cols (KOH ascending:
# 0.25 / 0.5 / 1.0 M). sharey='row' so each catalyst row has one symmetric y-axis
# sized to its own plotted data (tick magnitude 5% beyond max, rounded to nearest 10).
# y-tick marks/labels appear only on the leftmost column.

V_LOW_MAX = 0.4   # voltage threshold for the cycle-1 H-UPD slice

def _blank_c1_slice(d):
    """Cycle 1 ascending H-UPD tail: from V-minimum row through V <= V_LOW_MAX.
    Voltage-based — survives any row count or hardcoded-index edit."""
    imin = d['Ewe_RHE_corr'].idxmin()
    s    = d.loc[imin:]
    return s[s['Ewe_RHE_corr'] <= V_LOW_MAX]


def _round_to_10(x):
    return float(np.round(x / 10.0) * 10.0)


def plot_blank_cvs():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 3, figsize=(180*mm, 135*mm), sharey='row')
    tickfont, labelsize = 10, 10
    linewidth = 1
    kohs_asc = [250, 500, 1000]
    names, color = cat_names, cat_colors

    row_extents = [0.0] * 6   # per-row max |j| across the 3 KOH panels

    for row, cat in enumerate(catalysts):
        for col, koh in enumerate(kohs_asc):
            a = ax[row][col]
            d = all_run_data[koh][cat][0]['processed']['blank']
            c1_low = _blank_c1_slice(d)
            c1_low.plot(x='Ewe_RHE_corr', y='j1uA', c=color[row],
                        linestyle='solid', linewidth=linewidth, ax=a)
            if 'Ewe_RHE_corr2' in d.columns:
                d.plot(x='Ewe_RHE_corr2', y='j2uA', c=color[row],
                       linestyle='solid', linewidth=linewidth, ax=a)
            a.get_legend().remove()
            for sp in a.spines.values(): sp.set_visible(True)
            a.tick_params(axis='x', which='major', labelsize=tickfont)
            if row < 5:
                a.set_xticks([])
                a.get_xaxis().set_visible(False)
            else:
                a.set_xticks([0.1, 0.5, 0.9, 1.3])
                a.get_xaxis().set_visible(True)

            # Row-wide |j| extent from ONLY what's actually plotted
            j_concat = [c1_low['j1uA'].dropna().values]
            if 'j2uA' in d.columns:
                j_concat.append(d['j2uA'].dropna().values)
            j_all = np.concatenate(j_concat) if j_concat else np.array([0.0])
            if j_all.size:
                row_extents[row] = max(row_extents[row], float(np.nanmax(np.abs(j_all))))

    # Per-row symmetric ylim + inward tick (sharey='row' propagates to whole row).
    # Plot range preserved (~10% beyond data); tick magnitude moved inward to ~75% of
    # the range so labels never butt up against the panel edges where adjacent rows meet.
    for row in range(6):
        extent = row_extents[row]
        outer_mag = max(_round_to_10(extent * 1.05), 10.0)   # for ylim only
        ylim_mag  = outer_mag * 1.05                         # plot range (kept the same)
        tick_mag  = max(_round_to_10(ylim_mag * 0.75), 10.0) # ticks at ~75% of range
        if tick_mag >= ylim_mag:                             # guard: must fit inside
            tick_mag = _round_to_10(ylim_mag * 0.7)
        ax[row][0].set_ylim([-ylim_mag, ylim_mag])
        ax[row][0].set_yticks([-tick_mag, 0, tick_mag])
        ax[row][0].tick_params(axis='y', which='major', labelsize=tickfont)
        ax[row][0].tick_params(axis='y', which='minor', left=False, right=False)
        ax[row][0].minorticks_off()
        # Hide y-tick marks + labels on middle and right columns
        for col in (1, 2):
            ax[row][col].tick_params(axis='y', which='both',
                                     left=False, right=False, labelleft=False)

    # Top-row column titles
    ax[0][0].set_title(r'[KOH] = 0.25 M', fontsize=labelsize, fontweight='bold')
    ax[0][1].set_title(r'[KOH] = 0.5 M',  fontsize=labelsize, fontweight='bold')
    ax[0][2].set_title(r'[KOH] = 1.0 M',  fontsize=labelsize, fontweight='bold')

    # In-set catalyst labels (col 0, lower-left)
    for r in range(6):
        ax[r][0].text(0.05, 0.85, names[r], color=color[r],
                      fontsize=labelsize, rotation=0, va='center',
                      transform=ax[r][0].transAxes)

    # Single common x-label below the grid (avoids 3 overlapping per-panel labels)
    for col in range(3):
        ax[5][col].set_xlabel('', fontsize=labelsize, fontweight='bold')
    fig.text(0.515, 0.02, 'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')
    fig.text(0.04, 0.5, r"Current Density ($\mu$A/$cm^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    # wspace > 0 so adjacent xtick labels (1.3 ↔ 0.1) do not collide
    plt.subplots_adjust(wspace=0.15, hspace=0.0)
    decorate_figure(fig, 'Blank CVs (no CO)')
    plt.savefig(os.path.join(batch_root, 'Blank_CVs_SI.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_blank_cvs()


# In[13]:


# --- CO stripping figure — replicates CO_Stripping.ipynb cell 59 layout ---
# 6 rows (catalysts: Pd100 top → Ag90Pd10 bottom) × 3 cols (KOH ascending:
# 0.25 / 0.5 / 1.0 M). One representative run (RunA) per (catalyst, KOH).

def plot_co_stripping():
    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 3, figsize=(180*mm, 180*mm), sharex=True, sharey='row')
    tickfont, labelsize = 10, 10
    linewidth = 1
    kohs_asc = [250, 500, 1000]
    names, color = cat_names, cat_colors

    row_maxes = [0.0] * 6
    for row, cat in enumerate(catalysts):
        for col, koh in enumerate(kohs_asc):
            a = ax[row][col]
            d = all_run_data[koh][cat][0]['processed']['stripping']
            # Common forward (anodic) sweep — drop first 5 noisy points,
            # stop at the smaller idxmax of the two cycles
            imax = min(d['Ewe1c'].idxmax(), d['Ewe2c'].idxmax())
            anodic = d.loc[5:imax]
            anodic.plot(x='Ewe1c', y='j1uA', c=color[row],
                        linestyle='solid', linewidth=linewidth, ax=a)
            anodic.plot(x='Ewe2c', y='j2uA', c=color[row],
                        linestyle='dashed', linewidth=linewidth, ax=a)
            a.fill_between(anodic['Ewe1c'], anodic['j1uA'], anodic['j2uA'],
                           where=anodic['j1uA'] > anodic['j2uA'],
                           color=color[row], alpha=0.2)
            row_maxes[row] = max(row_maxes[row], float(anodic['j1uA'].max()))
            a.get_legend().remove()
            a.set_xticks([0.2, 0.5, 0.8, 1.1])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='both', which='major', labelsize=tickfont)
            a.set_xlim([0.15, 1.15])

    # Row-wide y-tick (single tick) — sharey='row' propagates; hide ticks on cols 1,2
    for row in range(6):
        ax[row][0].set_yticks([np.round(row_maxes[row] + 10, -1)])
        for col in (1, 2):
            ax[row][col].tick_params(axis='y', which='both',
                                     left=False, right=False, labelleft=False)

    # Top-row column titles: KOH ascending left→right
    ax[0][0].set_title('0.25 M KOH', fontsize=labelsize, fontweight='bold')
    ax[0][1].set_title('0.5 M KOH',  fontsize=labelsize, fontweight='bold')
    ax[0][2].set_title('1.0 M KOH',  fontsize=labelsize, fontweight='bold')

    # In-set catalyst names on column 2 (rightmost), upper-left of each panel
    textleftright, textupdown = 0.05, 0.85
    for r in range(6):
        ax[r][2].text(textleftright, textupdown, names[r], color=color[r],
                      fontsize=labelsize, rotation=0, va='center',
                      transform=ax[r][2].transAxes)

    # Center-bottom x-label only (others blank)
    for col in range(3):
        ax[5][col].set_xlabel('', fontsize=labelsize, fontweight='bold')
    fig.text(0.515, 0.02, 'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    fig.text(0.04, 0.5, r"Current Density ($\mu$A/$cm^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    decorate_figure(fig, 'CO stripping voltammetry (Run A)')
    plt.savefig(os.path.join(batch_root, 'CO_stripping_SI.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_co_stripping()


# In[14]:


def plot_cvs_runA_per_KOH(koh):
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 4, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1
    direction = 'in'
    pad = -30
    names, color = cat_names, cat_colors

    for row, cat in enumerate(catalysts):
        for col in range(4):
            a = ax[row][col]
            co_keys_asc = ['0pt1', '1', '10', '100']
            co = co_keys_asc[col]
            d   = all_run_data[koh][cat][0]['processed'][co]
            imx = d['Ewe_RHE_corr'].idxmax()
            fwd = d.loc[:imx]
            rev = d.loc[imx:]
            fwd.plot(x='Ewe_RHE_corr', y='juA', c=color[row],
                     linestyle='solid',  linewidth=linewidth, ax=a, legend=False)
            rev.plot(x='Ewe_RHE_corr', y='juA', c=color[row],
                     linestyle='dashed', linewidth=linewidth, ax=a, legend=False)
            a.set_xticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=(pad + 8) if col == 0 else pad)
    for col in range(4):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # Column ylim/yticks (verbatim from plot_SI_grouped)
    for col, (yl, yt) in enumerate([(250, 200), (1250, 1000), (2500, 2000), (4400, 3500)]):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[yt], ylim=[0, yl])

    # Catalyst labels (in-panel, col 0) — Pd100 uses x=0.7, others x=0.5
    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    # Suppress any per-panel xlabels
    for col in range(4):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')

    for col, t in enumerate([r'0.1% $P_{CO}$', r'1% $P_{CO}$', r'10% $P_{CO}$', r'100% $P_{CO}$']):
        ax[0][col].set_title(t, fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{RHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)

    # Panel letters
    start = 0.135
    inc = 0.198
    for k, L in enumerate('ABCD'):
        fig.text(start + inc*k, 0.895, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(0.475, 1.015)

    koh_lbl = {1000: '1.0 M', 500: '0.5 M', 250: '0.25 M'}[koh]
    decorate_figure(fig, f'Run A CVs in {koh_lbl} KOH')
    plt.savefig(os.path.join(batch_root, f'CVs_RunA_{koh}mM.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)

for _koh in (1000, 500, 250):
    plot_cvs_runA_per_KOH(_koh)


# In[15]:


def plot_cvs_CO_RHE():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 3, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1
    direction = 'in'
    pad = -30
    names, color = cat_names, cat_colors

    for row, cat in enumerate(catalysts):
        for col in range(3):
            a = ax[row][col]
            kohs_asc = [250, 500, 1000]
            koh = kohs_asc[col]
            co_alpha = [('100', 1), ('10', 0.666), ('1', 0.5), ('0pt1', 0.333)]
            proc = all_run_data[koh][cat][0]['processed']
            for co, alpha_val in co_alpha:
                d   = proc[co]
                imx = d['Ewe_RHE_corr'].idxmax()
                fwd = d.loc[:imx]
                rev = d.loc[imx:]
                fwd.plot(x='Ewe_RHE_corr', y='juA', c=color[row],
                         linestyle='solid',  linewidth=linewidth,
                         alpha=alpha_val, ax=a, legend=False)
                rev.plot(x='Ewe_RHE_corr', y='juA', c=color[row],
                         linestyle='dashed', linewidth=linewidth,
                         alpha=alpha_val, ax=a, legend=False)
            a.set_xticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=pad)
    for col in range(3):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # Column ylim/yticks — same magnitude across all KOH cols (panels overlay all 4 CO%)
    for col in range(3):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[3500], ylim=[0, 4400])

    # Catalyst labels (in-panel, col 0) — Pd100 uses x=0.7, others x=0.5
    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    # Suppress any per-panel xlabels
    for col in range(3):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')

    ax[0][0].set_title(r'[KOH] = 0.25 M', fontsize=labelsize, fontweight='bold')
    ax[0][1].set_title(r'[KOH] = 0.5 M',  fontsize=labelsize, fontweight='bold')
    ax[0][2].set_title(r'[KOH] = 1.0 M',  fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{RHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)

    # Panel letters
    start = 0.135
    inc = 0.264
    for k, L in enumerate('ABC'):
        fig.text(start + inc*k, 0.835, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(0.475, 1.015)

    handles = _legend_handles_alpha_line(
        [('100%', 1), ('10%', 0.666), ('1%', 0.5), ('0.1%', 0.333)])
    decorate_figure(fig, 'Run A CVs — CO partial pressure overlay',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'CVs_CO_overlay_RHE.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)

plot_cvs_CO_RHE()


# In[16]:


def plot_cvs_KOH_SHE():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 4, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1
    direction = 'in'
    pad = -30
    names, color = cat_names, cat_colors

    for row, cat in enumerate(catalysts):
        for col in range(4):
            a = ax[row][col]
            co_keys_asc = ['0pt1', '1', '10', '100']
            co = co_keys_asc[col]
            for koh, alpha_val in {1000: 1, 500: 0.5, 250: 0.25}.items():
                d   = all_run_data[koh][cat][0]['processed'][co]
                imx = d['Ewe_SHE_corr'].idxmax()
                fwd = d.loc[:imx]
                rev = d.loc[imx:]
                fwd.plot(x='Ewe_SHE_corr', y='juA', c=color[row],
                         linestyle='solid',  linewidth=linewidth,
                         alpha=alpha_val, ax=a, legend=False)
                rev.plot(x='Ewe_SHE_corr', y='juA', c=color[row],
                         linestyle='dashed', linewidth=linewidth,
                         alpha=alpha_val, ax=a, legend=False)
            a.set_xticks([-0.3, -0.05, 0.2])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=(pad + 8) if col == 0 else pad)
    for col in range(4):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # Column ylim/yticks (verbatim from plot_SI_grouped)
    for col, (yl, yt) in enumerate([(250, 200), (1250, 1000), (2500, 2000), (4400, 3500)]):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[yt], ylim=[0, yl])

    # Catalyst labels (in-panel, col 0) — Pd100 uses x=0.7, others x=0.5
    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    # Suppress any per-panel xlabels
    for col in range(4):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')

    for col, t in enumerate([r'0.1% $P_{CO}$', r'1% $P_{CO}$', r'10% $P_{CO}$', r'100% $P_{CO}$']):
        ax[0][col].set_title(t, fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{SHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)

    # Panel letters
    start = 0.135
    inc = 0.198
    for k, L in enumerate('ABCD'):
        fig.text(start + inc*k, 0.835, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(-0.35, 0.25)

    handles = _legend_handles_alpha_line(
        [('1.0 M', 1), ('0.5 M', 0.5), ('0.25 M', 0.25)])
    decorate_figure(fig, 'Run A CVs — KOH concentration overlay (V$_{SHE}$)',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'CVs_KOH_overlay_SHE.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)

plot_cvs_KOH_SHE()


# In[17]:


# --- SI_CVs_stats_CO_Grouped_<KOH>mM (250 / 500 / 1000) ---

def plot_SI_grouped(koh):
    # Data assembly: avg processed → template column aliases → filter 0.52–0.98 V RHE
    agg = {}
    for cat in catalysts:
        d = build_avg_processed(all_run_data[koh][cat])
        d = d[(d['Ewe_RHE_corr'] >= 0.52) & (d['Ewe_RHE_corr'] <= 0.98)].reset_index(drop=True)
        for co_t, co_l in zip(['pt1', '1', '10', '100'], ['0pt1', '1', '10', '100']):
            d[f'Ewe_RHE_{co_t}_mean'] = d['Ewe_RHE_corr']
            d[f'j{co_t}_mean']        = d[f'j_mean_{co_l}']
            d[f'j{co_t}_stddev']      = d[f'j_std_{co_l}']
        agg[cat] = d

    # Plot params (verbatim from template)
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 4, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1; linestyle = 'solid'
    marker = 'o'; markersize = 0; markevery = 1
    direction = 'in'; pad = -30
    alpha = 1; alpha2 = 0.05; plotevery = 1

    names = cat_names; color = cat_colors
    co_keys = ['pt1', '1', '10', '100']
    Ag90 = agg['Ag90Pd10']

    # Per-panel plotting (6 rows × 4 cols)
    for row, cat in enumerate(catalysts):
        d = agg[cat]
        for col, co_t in enumerate(co_keys):
            xcol, ycol, ecol = f'Ewe_RHE_{co_t}_mean', f'j{co_t}_mean', f'j{co_t}_stddev'
            sl = d.loc[0:d[xcol].idxmax():plotevery]
            sl.plot(x=xcol, y=ycol, c=color[row], linestyle=linestyle, linewidth=linewidth,
                    marker=marker, markersize=markersize, markevery=markevery,
                    alpha=alpha, ax=ax[row][col])
            sl.plot(x=xcol, y=ycol, yerr=ecol, c=color[row], linestyle=linestyle, linewidth=linewidth,
                    marker=marker, markersize=markersize, markevery=1,
                    alpha=alpha2, ax=ax[row][col])
            a = ax[row][col]
            a.get_legend().remove()
            a.set_xticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
            a.set_yticks([np.round(Ag90[ycol].max(), -1)])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=(pad + 8) if col == 0 else pad)
    for col in range(4):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # Column y-limits / yticks (verbatim values)
    for col, (yl, yt) in enumerate([(250, 200), (1250, 1000), (2500, 2000), (4400, 3500)]):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[yt], ylim=[0, yl])

    # Catalyst labels (col 0): Pd100 row uses x=0.7, others x=0.5
    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    for col in range(4):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')
    for col, t in enumerate([r'0.1% $P_{CO}$', r'1% $P_{CO}$', r'10% $P_{CO}$', r'100% $P_{CO}$']):
        ax[0][col].set_title(t, fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{RHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)
    start, inc = 0.135, 0.198
    for k, L in enumerate('ABCD'):
        fig.text(start + inc*k, 0.895, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(0.475, 1.015)

    koh_lbl = {1000: '1.0 M', 500: '0.5 M', 250: '0.25 M'}[koh]
    decorate_figure(fig, f'Mean CVs in {koh_lbl} KOH (mean ± SD)')
    plt.savefig(os.path.join(batch_root, f'SI_CVs_stats_CO_Grouped_{koh}mM.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


for _koh in (500, 250, 1000):
    plot_SI_grouped(_koh)


# In[18]:


def plot_mean_cvs_PCO_SHE():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 4, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1
    direction = 'in'
    pad = -30
    names, color = cat_names, cat_colors

    for row, cat in enumerate(catalysts):
        for col in range(4):
            a = ax[row][col]
            co_keys_asc = ['0pt1', '1', '10', '100']
            co = co_keys_asc[col]
            koh_ls = [(1000, 'solid'), (500, 'dashed'), (250, 'dotted')]
            for koh, ls in koh_ls:
                run_results = all_run_data[koh][cat]
                avg   = build_avg_processed(run_results)
                shift = float(np.mean([rr['SHE_shift'] for rr in run_results]))
                a.plot(avg['Ewe_RHE_corr'].values - shift, avg[f'j_mean_{co}'].values,
                       color=color[row], linestyle=ls, linewidth=linewidth)
            a.set_xticks([-0.3, -0.05, 0.2])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=(pad + 8) if col == 0 else pad)
    for col in range(4):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # Column ylim/yticks (verbatim from plot_SI_grouped)
    for col, (yl, yt) in enumerate([(250, 200), (1250, 1000), (2500, 2000), (4400, 3500)]):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[yt], ylim=[0, yl])

    # Catalyst labels (in-panel, col 0) — Pd100 uses x=0.7, others x=0.5
    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    # Suppress any per-panel xlabels
    for col in range(4):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')

    for col, t in enumerate([r'0.1% $P_{CO}$', r'1% $P_{CO}$', r'10% $P_{CO}$', r'100% $P_{CO}$']):
        ax[0][col].set_title(t, fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{SHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)

    # Panel letters
    start = 0.135
    inc = 0.198
    for k, L in enumerate('ABCD'):
        fig.text(start + inc*k, 0.835, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(-0.35, 0.25)

    handles = _legend_handles_linestyle(
        [('1.0 M', 'solid'), ('0.5 M', 'dashed'), ('0.25 M', 'dotted')])
    decorate_figure(fig, 'Mean CVs — KOH concentration overlay (V$_{SHE}$)',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'Mean_CVs_PCO_cols_SHE.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)

plot_mean_cvs_PCO_SHE()


def plot_mean_cvs_KOH_RHE():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(6, 3, figsize=(180*mm, 135*mm), sharex=True)
    tickfont = labelsize = 10
    linewidth = 1
    direction = 'in'
    pad = -30
    names, color = cat_names, cat_colors

    for row, cat in enumerate(catalysts):
        for col in range(3):
            a = ax[row][col]
            kohs_asc = [250, 500, 1000]
            koh = kohs_asc[col]
            run_results = all_run_data[koh][cat]
            avg   = build_avg_processed(run_results)
            rhe_x = avg['Ewe_RHE_corr'].values
            pco_ls = [('100', 'solid'), ('10', 'dashed'), ('1', 'dotted'), ('0pt1', 'dashdot')]
            for co, ls in pco_ls:
                a.plot(rhe_x, avg[f'j_mean_{co}'].values,
                       color=color[row], linestyle=ls, linewidth=linewidth)
            a.set_xticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='y', which='major', labelsize=tickfont, direction=direction,
                          pad=pad)
    for col in range(3):
        ax[5][col].tick_params(axis='x', which='major', labelsize=tickfont, direction='out')

    # All-cols ylim — panels overlay all 4 CO%, so use 100% CO range
    for col in range(3):
        for row in range(6):
            plt.setp(ax[row][col], yticks=[3500], ylim=[0, 4400])

    ax[0][0].text(0.7, 0.85, names[0], color=color[0], fontsize=labelsize,
                  rotation=0, va='center', transform=ax[0][0].transAxes)
    for r in range(1, 6):
        ax[r][0].text(0.5, 0.85, names[r], color=color[r], fontsize=labelsize,
                      rotation=0, va='center', transform=ax[r][0].transAxes)

    for col in range(3):
        ax[5][col].set_xlabel('', fontsize=12, fontweight='bold')

    ax[0][0].set_title(r'[KOH] = 0.25 M', fontsize=labelsize, fontweight='bold')
    ax[0][1].set_title(r'[KOH] = 0.5 M',  fontsize=labelsize, fontweight='bold')
    ax[0][2].set_title(r'[KOH] = 1.0 M',  fontsize=labelsize, fontweight='bold')

    plt.minorticks_off()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    fig.text(0.515, 0.04, 'Potential (V$_{RHE}$)', ha='center', fontsize=labelsize)
    fig.text(0.08, 0.5, r"Current Density ($\mu$A/$cm_{Pd}^{2}$)",
             va='center', rotation='vertical', fontsize=labelsize)

    start = 0.135
    inc = 0.264
    for k, L in enumerate('ABC'):
        fig.text(start + inc*k, 0.835, f'{L})', ha='center', fontsize=labelsize)

    for axi in ax.flat:
        axi.set_xlim(0.475, 1.015)

    handles = _legend_handles_linestyle(
        [('100%', 'solid'), ('10%', 'dashed'),
         ('1%', 'dotted'), ('0.1%', 'dashdot')])
    decorate_figure(fig, 'Mean CVs — CO partial pressure overlay (V$_{RHE}$)',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'Mean_CVs_KOH_cols_RHE.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_mean_cvs_KOH_RHE()


# In[19]:


# --- alloy_Logj figure: identical layout to SI Log Fig.ipynb cell 3, with our data ---
# 4 rows (CO% desc: 100, 10, 1, 0.1) × 3 cols (KOH asc: 0.25, 0.5, 1.0 M).
# Each panel: 6 catalysts as colored mean lines with std error bars.



def plot_log_j_fig():
    mpl.rcParams['figure.dpi'] = 1200
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(4, 3, figsize=(180*mm, 135*mm), sharex=True, sharey=True)
    tickfont, labelsize = 10, 8
    linewidth = 1
    markersize = 0
    markevery = 10
    errorevery = 10
    direction = 'in'
    linestyle = 'solid'

    avg = {koh: {cat: build_avg_logj(all_run_data[koh][cat]) for cat in catalysts}
           for koh in (1000, 500, 250)}

    # Columns ascending KOH (c=250, b=500, a=1000)
    koh_by_col   = [250, 500, 1000]
    # Rows descending CO% (100, 10, 1, 0.1%)
    co_keys_desc = ['100', '10', '1', '0pt1']

    for col, koh in enumerate(koh_by_col):
        for row, co in enumerate(co_keys_desc):
            a = ax[row][col]
            for cat_idx, cat in enumerate(catalysts):
                d = avg[koh][cat]
                d.plot(x='Ewe_RHE_corr', y=f'Logj_mean_{co}',
                       yerr=f'Logj_std_{co}',
                       c=cat_colors[cat_idx], linestyle=linestyle,
                       linewidth=linewidth, marker='o',
                       markersize=markersize, markevery=markevery,
                       errorevery=errorevery, ax=a, legend=False)
            a.set_xlim([0.55, 0.95])
            a.set_ylim([-2, 12])
            a.set_xticks([0.6, 0.75, 0.9])
            a.set_yticks([0, 2, 4, 6, 8, 10])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.set_xlabel('')
            a.set_ylabel('')
            a.tick_params(axis='both', which='major', labelsize=tickfont, direction=direction)



    # Top-row column titles: KOH ascending
    OH_concentrations = ['1.0 M', '0.5 M', '0.25 M']
    ax[0][0].set_title(OH_concentrations[2] + ' KOH', fontsize=labelsize, fontweight='bold')
    ax[0][1].set_title(OH_concentrations[1] + ' KOH', fontsize=labelsize, fontweight='bold')
    ax[0][2].set_title(OH_concentrations[0] + ' KOH', fontsize=labelsize, fontweight='bold')


    # Per-row CO% y-labels on col 0 — short subscript style, smaller font
    co_short = [r"$\ln|j$ ($\mu$A/$cm_{Pd}^{2}$)$|_{100\%}$",
                r"$\ln|j$ ($\mu$A/$cm_{Pd}^{2}$)$|_{10\%}$",
                r"$\ln|j$ ($\mu$A/$cm_{Pd}^{2}$)$|_{1\%}$",
                r"$\ln|j$ ($\mu$A/$cm_{Pd}^{2}$)$|_{0.1\%}$"]
    for r, lbl in enumerate(co_short):
        ax[r][0].set_ylabel(lbl, fontsize=9, fontweight='bold')

    plt.subplots_adjust(wspace=0, hspace=0.0)
    plt.minorticks_off()

    start, inc = 0.135, 0.26
    fig.text(start, 0.895, 'A)', ha='center', fontsize=labelsize)
    fig.text(start + inc, 0.895, 'B)', ha='center', fontsize=labelsize)
    fig.text(start + inc*2, 0.895, 'C)', ha='center', fontsize=labelsize)
    fig.text(0.515, 0.02, 'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    decorate_figure(fig, r'$\ln|j$ ($\mu$A/$cm_{Pd}^{2}$)$|$ across alloys')
    plt.savefig(os.path.join(batch_root, 'alloy_Logj.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_log_j_fig()


# In[20]:


# --- Alpha OH-dependence figure ---
# Rows: 4 CO% (top→bottom: 100, 10, 1, 0.1%).  Cols: 6 catalysts (Pd100 left → Ag90Pd10 right).
# Each panel overlays 3 KOH traces (1000/500/250 mM) differentiated by transparency.

def plot_alpha_grouped():
    avg = {koh: {cat: build_avg_alpha(all_run_data[koh][cat]) for cat in catalysts}
           for koh in (1000, 500, 250)}

    # Plot params: match current-density (SI grouped) plots — bold fonts, high DPI
    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(4, 6, figsize=(180*mm, 135*mm), sharex=True, sharey='row')
    tickfont, labelsize, pad = 8, 10, 3
    linewidth, marker, markersize = 1, 'o', 4
    markevery2 = errorevery2 = 40
    capsize    = markersize / 2
    direction  = 'in'
    koh_alpha  = {1000: 1, 500: 0.5, 250: 0.25}     # alpha1/2/3
    co_keys_desc = ['100', '10', '1', '0pt1']        # rows: CO% descending
    names, color = cat_names, cat_colors

    for row, co in enumerate(co_keys_desc):
        for col, cat in enumerate(catalysts):
            a = ax[row][col]
            for koh, alpha_val in koh_alpha.items():
                d  = avg[koh][cat].copy()
                shift = float(np.mean([rr['SHE_shift'] for rr in all_run_data[koh][cat]]))
                d['Ewe_SHE_corr'] = d['Ewe_RHE_corr'] - shift
                sl = d[(d['Ewe_RHE_corr'] >= 0.6) & (d['Ewe_RHE_corr'] <= 1)]
                sl.plot(x='Ewe_SHE_corr', y=f'alpha_mean_{co}',
                        yerr=f'alpha_std_{co}', c=color[col],
                        linestyle='none', linewidth=linewidth,
                        errorevery=errorevery2, markevery=markevery2,
                        alpha=alpha_val, capsize=capsize, ax=a,
                        marker=marker, markersize=markersize,
                        markeredgecolor='black', markeredgewidth=0.3)
            a.get_legend().remove()
            a.set_xticks([-0.2, 0, 0.2])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='both', which='major', labelsize=tickfont,
                          direction=direction, pad=pad)
            a.set_xlim([-0.30, 0.25])
            a.set_ylim([-0.6, 0.6])
            a.set_yticks([-0.5, 0, 0.5])

    # Top-row column titles: catalyst names (colored)
    for col, n in enumerate(names):
        ax[0][col].set_title(n, fontsize=labelsize, fontweight='bold', color=color[col])
    # Bottom-row x-labels blank (we use fig.text for common label)
    for col in range(6):
        ax[-1][col].set_xlabel('', fontsize=labelsize, fontweight='bold')

    # Left-column y-labels: α with P_CO subscript (one per row, CO% descending)
    pco_labels = [r"$\alpha_{100\%\,P_{CO}}$",
                  r"$\alpha_{10\%\,P_{CO}}$",
                  r"$\alpha_{1\%\,P_{CO}}$",
                  r"$\alpha_{0.1\%\,P_{CO}}$"]
    for r, lbl in enumerate(pco_labels):
        ax[r][0].set_ylabel(lbl, fontsize=labelsize, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.text(0.52, 0.05, 'Potential (V$_{SHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    handles = _legend_handles_alpha_marker(
        [('1.0 M', 1), ('0.5 M', 0.5), ('0.25 M', 0.25)],
        marker=marker, markersize=markersize)
    decorate_figure(fig, r'Transfer coefficient $\alpha$ — OH$^-$ dependence',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'alpha_overall_OHdependence.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_alpha_grouped()


# In[21]:


# --- Alpha CO-dependence figure ---
# Rows: 3 KOH (top→bottom: 1.0, 0.5, 0.25 M).  Cols: 6 catalysts (Pd100 left → Ag90Pd10 right).
# Each panel overlays 4 CO% traces (100/10/1/0.1%) differentiated by transparency.

def plot_alpha_co_grouped():
    avg = {koh: {cat: build_avg_alpha(all_run_data[koh][cat]) for cat in catalysts}
           for koh in (1000, 500, 250)}

    # Plot params: match current-density (SI grouped) plots — bold fonts, high DPI
    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(3, 6, figsize=(180*mm, 115*mm), sharex=True, sharey='row')
    tickfont, labelsize, pad = 8, 10, 3
    linewidth, marker, markersize = 1, 'o', 4
    markevery2 = errorevery2 = 40
    capsize    = markersize / 2
    direction  = 'in'
    co_alpha   = [('100', 1), ('10', 0.666), ('1', 0.5), ('0pt1', 0.333)]
    kohs_desc  = [1000, 500, 250]                  # rows: KOH descending
    names, color = cat_names, cat_colors

    for row, koh in enumerate(kohs_desc):
        for col, cat in enumerate(catalysts):
            a  = ax[row][col]
            d  = avg[koh][cat]
            sl = d[(d['Ewe_RHE_corr'] >= 0.6) & (d['Ewe_RHE_corr'] <= 1)]
            for co, alpha_val in co_alpha:
                sl.plot(x='Ewe_RHE_corr', y=f'alpha_mean_{co}',
                        yerr=f'alpha_std_{co}', c=color[col],
                        linestyle='none', linewidth=linewidth,
                        errorevery=errorevery2, markevery=markevery2,
                        alpha=alpha_val, capsize=capsize, ax=a,
                        marker=marker, markersize=markersize,
                        markeredgecolor='black', markeredgewidth=0.3)
            a.get_legend().remove()
            a.set_xticks([0.6, 0.8, 1.0])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='both', which='major', labelsize=tickfont,
                          direction=direction, pad=pad)
            a.set_xlim([0.55, 1.05])
            a.set_ylim([-0.6, 0.6])
            a.set_yticks([-0.5, 0, 0.5])

    # Top-row column titles: catalyst names (colored)
    for col, n in enumerate(names):
        ax[0][col].set_title(n, fontsize=labelsize, fontweight='bold', color=color[col])
    for col in range(6):
        ax[-1][col].set_xlabel('', fontsize=labelsize, fontweight='bold')

    # Left-column y-labels: α with KOH subscript (one per row)
    koh_labels = [r"$\alpha_{1.0\,M\,KOH}$",
                  r"$\alpha_{0.5\,M\,KOH}$",
                  r"$\alpha_{0.25\,M\,KOH}$"]
    for r, lbl in enumerate(koh_labels):
        ax[r][0].set_ylabel(lbl, fontsize=labelsize, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.text(0.52, 0.05, 'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    handles = _legend_handles_alpha_marker(
        [('100%', 1), ('10%', 0.666), ('1%', 0.5), ('0.1%', 0.333)],
        marker=marker, markersize=markersize)
    decorate_figure(fig, r'Transfer coefficient $\alpha$ — CO dependence',
                    handles=handles)
    plt.savefig(os.path.join(batch_root, 'alpha_overall_COdependence.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_alpha_co_grouped()


# In[22]:


# --- CO reaction-order figure (per adjacent CO% step pair, KOH × catalysts) ---
# Layout mirrors plot_alpha_co_grouped: 3 rows (KOH descending 1.0→0.25 M)
# × 6 cols (catalysts, Pd100 left → Ag90Pd10 right).
# Each panel overlays 3 step-pair slopes (0.1-1, 1-10, 10-100%) via alpha gradient.

def plot_co_order_grouped():
    avg = {koh: {cat: build_co_order_steps(all_run_data[koh][cat]) for cat in catalysts}
           for koh in (1000, 500, 250)}

    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(3, 6, figsize=(180*mm, 115*mm), sharex=True, sharey='row')
    tickfont, labelsize, pad = 8, 10, 3
    linewidth, marker, markersize = 1, 'o', 4
    markevery2 = errorevery2 = 40
    capsize    = markersize / 2
    direction  = 'in'
    # Pair-alpha gradient: 10-100% darkest, 0.1-1% lightest
    pair_alpha = [(('10', '100'), 1), (('1', '10'), 0.5), (('0pt1', '1'), 0.25)]
    kohs_desc  = [1000, 500, 250]
    names, color = cat_names, cat_colors

    for row, koh in enumerate(kohs_desc):
        for col, cat in enumerate(catalysts):
            a  = ax[row][col]
            d  = avg[koh][cat]
            sl = d[(d['Ewe_RHE_corr'] >= 0.6) & (d['Ewe_RHE_corr'] <= 1)]
            for (lo, hi), alpha_val in pair_alpha:
                sl.plot(x='Ewe_RHE_corr', y=f'CO_Order_{lo}_{hi}',
                        yerr=f'CO_Order_{lo}_{hi}_err', c=color[col],
                        linestyle='none', linewidth=linewidth,
                        errorevery=errorevery2, markevery=markevery2,
                        alpha=alpha_val, capsize=capsize, ax=a,
                        marker=marker, markersize=markersize,
                        markeredgecolor='black', markeredgewidth=0.3)
            a.get_legend().remove()
            a.set_xticks([0.6, 0.8, 1.0])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='both', which='major', labelsize=tickfont,
                          direction=direction, pad=pad)
            a.set_xlim([0.55, 1.05])
            a.set_ylim([0, 1])
            a.set_yticks([0, 0.5, 1.0])

    for col, n in enumerate(names):
        ax[0][col].set_title(n, fontsize=labelsize, fontweight='bold', color=color[col])
    for col in range(6):
        ax[-1][col].set_xlabel('', fontsize=labelsize, fontweight='bold')

    # Left-column y-labels: n_CO with KOH subscript (one per row)
    koh_labels = [r"$\delta_{CO,\,1.0\,M\,KOH}$",
                  r"$\delta_{CO,\,0.5\,M\,KOH}$",
                  r"$\delta_{CO,\,0.25\,M\,KOH}$"]
    for r, lbl in enumerate(koh_labels):
        ax[r][0].set_ylabel(lbl, fontsize=labelsize, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.text(0.52, 0.05, 'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    handles = _legend_handles_alpha_marker(
        [('10–100%', 1), ('1–10%', 0.5), ('0.1–1%', 0.25)],
        marker=marker, markersize=markersize)
    decorate_figure(fig, 'CO reaction order (grouped by KOH)', handles=handles)
    plt.savefig(os.path.join(batch_root, 'CO_reaction_order.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_co_order_grouped()


# In[23]:


# --- CO reaction-order figure grouped by CATALYST ---
# Layout: 3 rows (CO step pairs, top→bottom: 10→100%, 1→10%, 0.1→1%) × 6 cols
# (catalysts, Pd100 left → Ag90Pd10 right). Each panel overlays 3 KOH traces
# (1.0/0.5/0.25 M) differentiated by transparency (1.0 M darkest, 0.25 M lightest).

def plot_co_order_KOH_dependence():
    avg = {koh: {cat: build_co_order_steps(all_run_data[koh][cat]) for cat in catalysts}
           for koh in (1000, 500, 250)}

    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(3, 6, figsize=(180*mm, 115*mm), sharex=True, sharey='row')
    tickfont, labelsize, pad = 8, 10, 3
    linewidth, marker, markersize = 1, 'o', 4
    markevery2 = errorevery2 = 40
    capsize    = markersize / 2
    direction  = 'in'
    # KOH transparency gradient: 1.0 M darkest → 0.25 M lightest
    koh_alpha = {1000: 1, 500: 0.5, 250: 0.25}
    # Rows: CO step pairs TOP→BOTTOM (10-100 top, 0.1-1 bottom)
    pairs_desc = [('10', '100'), ('1', '10'), ('0pt1', '1')]
    names, color = cat_names, cat_colors

    for row, (lo, hi) in enumerate(pairs_desc):
        for col, cat in enumerate(catalysts):
            a = ax[row][col]
            for koh, alpha_val in koh_alpha.items():
                d  = avg[koh][cat].copy()
                shift = float(np.mean([rr['SHE_shift'] for rr in all_run_data[koh][cat]]))
                d['Ewe_SHE_corr'] = d['Ewe_RHE_corr'] - shift
                sl = d[(d['Ewe_RHE_corr'] >= 0.6) & (d['Ewe_RHE_corr'] <= 1)]
                sl.plot(x='Ewe_SHE_corr', y=f'CO_Order_{lo}_{hi}',
                        yerr=f'CO_Order_{lo}_{hi}_err', c=color[col],
                        linestyle='none', linewidth=linewidth,
                        errorevery=errorevery2, markevery=markevery2,
                        alpha=alpha_val, capsize=capsize, ax=a,
                        marker=marker, markersize=markersize,
                        markeredgecolor='black', markeredgewidth=0.3)
            a.get_legend().remove()
            a.set_xticks([-0.2, 0, 0.2])
            for sp in a.spines.values(): sp.set_visible(True)
            a.get_xaxis().set_visible(True)
            a.tick_params(axis='both', which='major', labelsize=tickfont,
                          direction=direction, pad=pad)
            a.set_xlim([-0.30, 0.25])
            a.set_ylim([0, 1])
            a.set_yticks([0, 0.5, 1.0])

    # Top-row column titles: catalyst names (colored)
    for col, n in enumerate(names):
        ax[0][col].set_title(n, fontsize=labelsize, fontweight='bold', color=color[col])
    for col in range(6):
        ax[-1][col].set_xlabel('', fontsize=labelsize, fontweight='bold')

    # Left-column y-labels: δ_CO with step-pair subscript (one per row, matching top-to-bottom)
    step_labels = [r"$\delta_{CO}$ (10% → 100%)",
                   r"$\delta_{CO}$ (1% → 10%)",
                   r"$\delta_{CO}$ (0.1% → 1%)"]
    for r, lbl in enumerate(step_labels):
        ax[r][0].set_ylabel(lbl, fontsize=labelsize, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.text(0.52, 0.05, 'Potential (V$_{SHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    handles = _legend_handles_alpha_marker(
        [('1.0 M', 1), ('0.5 M', 0.5), ('0.25 M', 0.25)],
        marker=marker, markersize=markersize)
    decorate_figure(fig, 'CO reaction order (grouped by CO% regime)', handles=handles)
    plt.savefig(os.path.join(batch_root, 'CO_reaction_order_KOH_dependence.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_co_order_KOH_dependence()


# In[24]:


# --- OH reaction-order figure (global slope across all 3 KOH) ---
# Layout: 1 row × 6 cols (catalysts). 4 CO% overlaid per panel using the
# same alpha gradient as the CO-dependence alpha plot.

def plot_oh_order_grouped():
    avg = {cat: build_oh_order_global({koh: all_run_data[koh][cat]
                                       for koh in (1000, 500, 250)})
           for cat in catalysts}

    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(1, 6, figsize=(180*mm, 65*mm), sharex=True, sharey=True)
    tickfont, labelsize, pad = 8, 10, 3
    linewidth, marker, markersize = 1, 'o', 4
    markevery2 = errorevery2 = 40
    capsize    = markersize / 2
    direction  = 'in'
    # CO% transparency gradient: 100% darkest → 0.1% lightest (matches alpha CO-dep)
    co_alpha = [('100', 1), ('10', 0.666), ('1', 0.5), ('0pt1', 0.333)]
    names, color = cat_names, cat_colors

    for col, cat in enumerate(catalysts):
        a  = ax[col]
        d  = avg[cat]
        sl = d[(d['Ewe_SHE_corr'] >= -0.20) & (d['Ewe_SHE_corr'] <= 0.15)]
        for co, alpha_val in co_alpha:
            sl.plot(x='Ewe_SHE_corr', y=f'OH_Order_{co}',
                    yerr=f'OH_Order_{co}_err', c=color[col],
                    linestyle='none', linewidth=linewidth,
                    errorevery=errorevery2, markevery=markevery2,
                    alpha=alpha_val, capsize=capsize, ax=a,
                    marker=marker, markersize=markersize,
                    markeredgecolor='black', markeredgewidth=0.3)
        a.get_legend().remove()
        a.set_xticks([-0.2, 0, 0.2])
        for sp in a.spines.values(): sp.set_visible(True)
        a.get_xaxis().set_visible(True)
        a.tick_params(axis='both', which='major', labelsize=tickfont,
                      direction=direction, pad=pad)
        a.set_xlim([-0.25, 0.20])
        a.set_ylim([-0.5, 1.5])
        a.set_yticks([0, 0.5, 1.0])
        a.set_title(names[col], fontsize=labelsize, fontweight='bold', color=color[col])
        a.set_xlabel('', fontsize=labelsize, fontweight='bold')

    # Leftmost-column y-label
    ax[0].set_ylabel(r"$\delta_{OH}$", fontsize=labelsize, fontweight='bold')

    plt.subplots_adjust(wspace=0.15)
    fig.text(0.52, -0.05, 'Potential (V$_{SHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')

    handles = _legend_handles_alpha_marker(
        [('100%', 1), ('10%', 0.666), ('1%', 0.5), ('0.1%', 0.333)],
        marker=marker, markersize=markersize)
    decorate_figure(fig, 'OH reaction order across alloys',
                    handles=handles, bbox_y=0.90, top=0.65)
    plt.savefig(os.path.join(batch_root, 'OH_reaction_order.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_oh_order_grouped()


# --- Ag100 (pure Ag) summary — Run A, 0.5 M KOH ---
# 3 panels (left → right): Blank CV, CO stripping, CV forward sweeps (4 CO%).

def plot_ag100_summary():
    cat = 'Ag100'
    koh = 500
    if koh not in all_run_data or cat not in all_run_data[koh]:
        print(f'Ag100 / {koh} mM not present — skipping')
        return

    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams["font.weight"] = "bold"
    mm = 1/25.4
    fig, ax = plt.subplots(1, 3, figsize=(180*mm, 65*mm))
    labelsize = 10
    linewidth = 1
    ag_color  = ag_cat_colors[0]
    ag_name   = ag_cat_names[0]

    proc = all_run_data[koh][cat][0]['processed']

    # Local inline of the blank H-UPD slice (cycle 1 ascending tail
    # from V-minimum row through V <= 0.4 V) so this block stands alone.
    V_LOW_MAX = 0.4
    def _c1_low(d):
        imin = d['Ewe_RHE_corr'].idxmin()
        s    = d.loc[imin:]
        return s[s['Ewe_RHE_corr'] <= V_LOW_MAX]

    # Panel A — Blank CV (H-UPD slice of cycle 1 + cycle 2)
    aA = ax[0]
    d = proc['blank']
    c1_low = _c1_low(d)
    c1_low.plot(x='Ewe_RHE_corr', y='ImuA', c=ag_color,
                linestyle='solid', linewidth=linewidth, ax=aA, legend=False)
    if 'Ewe_RHE_corr2' in d.columns:
        # cycle 2 raw current (µA) — process_run doesn't precompute this column
        aA.plot(d['Ewe_RHE_corr2'], d['I_2'] * 1000,
                color=ag_color, linestyle='solid', linewidth=linewidth)
    aA.set_title('Blank CV', fontsize=labelsize, fontweight='bold')
    aA.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    aA.set_xlim([0, 1.35])
    aA.set_yticks([-300, -200, -100, 0, 100])
    aA.set_ylim(-400, 200)
    aA.set_xlabel(''); aA.set_ylabel('')

    # Panel B — CO stripping (two complete cycles, IR-corrected, full forward + reverse sweep)
    # Per user: do NOT use the standard CO-stripping plot style (forward-sweep + fill)
    # for Ag100; show the raw two-cycle CVs instead.
    aB = ax[1]
    d = proc['stripping']
    aB.plot(d['Ewe1c'], d['I_1'] * 1000, color=ag_color,
            linestyle='solid',  linewidth=linewidth, label='')
    aB.plot(d['Ewe2c'], d['I_2'] * 1000, color=ag_color,
            linestyle='dashed', linewidth=linewidth, label='')
    aB.legend(fontsize=8, frameon=False, loc='upper left')
    aB.set_title('CO stripping', fontsize=labelsize, fontweight='bold')
    aB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    aB.set_xlim([0, 1.2])
    aB.set_yticks([-60, -30, 0, 30, 60])
    aB.set_ylim(-60, 60)
    aB.set_xlabel(''); aB.set_ylabel('')

    # Panel C — CV forward sweeps overlaid (4 CO%, linestyle-encoded:
    # solid/dashed/dotted/dashdot — same convention as Mean_CVs_KOH_cols_RHE).
    aC = ax[2]
    pco_ls = [('100',  '100% CO', 'solid'),
              ('10',   '10% CO',  'dashed'),
              ('1',    '1% CO',   'dotted'),
              ('0pt1', '0.1% CO', 'dashdot')]
    for co, lbl, ls in pco_ls:
        d = proc[co]
        imx = d['Ewe_RHE_corr'].idxmax()
        fwd = d.loc[:imx]
        aC.plot(fwd['Ewe_RHE_corr'], fwd['ImuA'], color=ag_color,
                linestyle=ls, linewidth=linewidth, label=lbl)
    aC.legend(fontsize=8,
              frameon=False, loc='upper left')
    aC.set_title('CV forward sweeps', fontsize=labelsize, fontweight='bold')
    aC.set_ylim(-2, 6)
    aC.set_xlabel(''); aC.set_ylabel('')

    for a in ax:
        a.tick_params(axis='both', labelsize=labelsize)
        for sp in a.spines.values(): sp.set_visible(True)

    plt.minorticks_off()
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.78, wspace=0.25)
    fig.text(0.54, 0.04, r'Potential (V$_{RHE}$)', ha='center',
             fontsize=labelsize, fontweight='bold')
    fig.text(0.02, 0.5, r"Current ($\mu$A)",
             va='center', rotation='vertical', fontsize=labelsize, fontweight='bold')
    decorate_figure(fig, f'{ag_name} (Run A, 0.5 M KOH)', suptitle_y=0.95)
    plt.savefig(os.path.join(batch_root, 'Ag100_summary_RunA_500mM.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=1200)


plot_ag100_summary()


# --- Second-derivatives analysis: alpha and delta_OH vs potential ---------------
#
# For each (catalyst, CO%, KOH) panel, plot the transfer coefficient (alpha)
# and OH reaction order (delta_OH) vs E (V vs SHE). A linear regression
# inside the kinetic window E <= E_20pct returns two summary slopes per panel:
#     d(alpha)/dE     (V^-1)
#     d(delta_OH)/dE  (V^-1)
# Because alpha = d(ln|j|)/dE * RT/F and delta_OH = d(log|j|)/d(log[OH-])
# are themselves first derivatives of log|j|, these slopes are effectively
# second derivatives of the log-current response.
#
# Data sources:
#   - alpha       : build_avg_alpha (per cat / KOH on the commonRHE grid),
#                   shifted to V_SHE by subtracting the per-KOH SHE_shift.
#   - delta_OH    : build_oh_order_global (per cat on the commonSHE grid);
#                   one curve per catalyst, shared across the 3 KOH panels.
#   - E_20pct     : build_avg_logj — geometric-mean |j| across runs, taken
#                   where it reaches 50% of its anodic-sweep maximum,
#                   linearly interpolated and shifted RHE -> SHE.

from matplotlib.ticker import FuncFormatter as _FuncFormatter

PRESSURES_SD = [
    ('0pt1', '0.1% CO'),
    ('1',    '1% CO'),
    ('10',   '10% CO'),
    ('100',  '100% CO'),
]
KOH_KEYS_ASC = [250, 500, 1000]
KOH_TITLES_SD = {250: '[KOH] = 0.25 M',
                 500: '[KOH] = 0.5 M',
                 1000: '[KOH] = 1 M'}

WINDOW_COLOR = '#FFA500'
N_ANALYSIS_PTS = 20

# Analysis potentials: 20 evenly-spaced points in V vs RHE per row, with a
# catalyst-dependent range chosen to span each catalyst's kinetic region:
#   Pd-rich  (Pd100, Ag10Pd90)            : RHE [0.750, 0.950]
#   High-Ag  (Ag25Pd75 ... Ag90Pd10)      : RHE [0.600, 0.950]
RHE_RANGE_BY_ROW = {
    0: (0.750, 0.950),  # Pd100
    1: (0.750, 0.950),  # Ag10Pd90
    2: (0.600, 0.950),  # Ag25Pd75
    3: (0.600, 0.950),  # Ag50Pd50
    4: (0.600, 0.950),  # Ag75Pd25
    5: (0.600, 0.950),  # Ag90Pd10
}


def _mean_she_shift(cat, koh_mM):
    runs = all_run_data.get(koh_mM, {}).get(cat)
    if not runs:
        return np.nan
    return float(np.mean([rr['SHE_shift'] for rr in runs]))


def compute_e20pct_she(cat, co_label, koh_mM):
    """E (V vs SHE) where geometric-mean |j| reaches 50% of its maximum
    on the anodic sweep. Returns np.nan if threshold never crossed."""
    runs = all_run_data.get(koh_mM, {}).get(cat)
    if not runs:
        return np.nan
    avg_logj = build_avg_logj(runs)
    logj     = avg_logj[f'Logj_mean_{co_label}'].values
    E_rhe    = avg_logj['Ewe_RHE_corr'].values
    mask = np.isfinite(logj)
    if not mask.any():
        return np.nan
    j = np.exp(logj[mask])
    E_r = E_rhe[mask]
    thresh = 0.50 * j.max()
    above = j >= thresh
    if not above.any():
        return np.nan
    first = int(np.argmax(above))
    if first == 0:
        e_rhe = float(E_r[0])
    else:
        E0, j0 = E_r[first - 1], j[first - 1]
        E1, j1 = E_r[first],     j[first]
        e_rhe = float(E0 + (thresh - j0) / (j1 - j0) * (E1 - E0)
                      if j1 != j0 else E1)
    she_shift = _mean_she_shift(cat, koh_mM)
    return e_rhe - she_shift


def linreg_window_sd(E, y, e20_she, min_points=3, robust=False):
    """Trendline of y vs E inside the kinetic window (E <= e20_she). With
    robust=True, fits by Theil-Sen (median of all pairwise slopes). Returns
    (slope, intercept, E_w, y_fit), or None if fewer than min_points fall in
    the window."""
    mask = (E <= e20_she) & np.isfinite(y)
    E_w, y_w = E[mask], y[mask]
    if len(E_w) < min_points:
        return None
    if robust:
        slope, intercept, _, _ = theilslopes(y_w, E_w)
    else:
        A = np.column_stack([E_w, np.ones(len(E_w))])
        coeffs, _, _, _ = np.linalg.lstsq(A, y_w, rcond=None)
        slope, intercept = coeffs
    return slope, intercept, E_w, slope * E_w + intercept


def _sample_at(E_grid, y_grid, E_target, ystd_grid=None):
    n = len(E_target)
    mask = np.isfinite(y_grid)
    if not mask.any():
        return np.full(n, np.nan), np.full(n, np.nan)
    fi = interp1d(E_grid[mask], y_grid[mask], kind='linear',
                  bounds_error=False, fill_value=np.nan)
    y_t = fi(E_target)
    if ystd_grid is None:
        return y_t, np.full(n, np.nan)
    smask = np.isfinite(ystd_grid)
    if smask.any():
        fs = interp1d(E_grid[smask], ystd_grid[smask], kind='linear',
                      bounds_error=False, fill_value=np.nan)
        return y_t, fs(E_target)
    return y_t, np.full(n, np.nan)


def _load_alpha_oh_for_panel(cat, ri, co_label, koh_mM):
    n = N_ANALYSIS_PTS
    rhe_lo, rhe_hi = RHE_RANGE_BY_ROW[ri]
    E_rhe_target = np.linspace(rhe_lo, rhe_hi, n)

    runs = all_run_data.get(koh_mM, {}).get(cat)
    if not runs:
        she_shift = 0.808 if koh_mM == 500 else (0.826 if koh_mM == 1000 else 0.790)
        return (E_rhe_target - she_shift,
                np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, np.nan), np.full(n, np.nan))

    she_shift = _mean_she_shift(cat, koh_mM)
    E_she_target = E_rhe_target - she_shift

    avg_a = build_avg_alpha(runs)
    alpha, alpha_err = _sample_at(
        avg_a['Ewe_RHE_corr'].values,
        avg_a[f'alpha_mean_{co_label}'].values,
        E_rhe_target,
        avg_a[f'alpha_std_{co_label}'].values,
    )

    koh_to_runs = {koh: all_run_data[koh][cat]
                   for koh in (1000, 500, 250)
                   if cat in all_run_data.get(koh, {})}
    if len(koh_to_runs) >= 2:
        avg_oh = build_oh_order_global(koh_to_runs)
        delta_oh, delta_oh_err = _sample_at(
            avg_oh['Ewe_SHE_corr'].values,
            avg_oh[f'OH_Order_{co_label}'].values,
            E_she_target,
            avg_oh[f'OH_Order_{co_label}_err'].values,
        )
    else:
        delta_oh    = np.full(n, np.nan)
        delta_oh_err = np.full(n, np.nan)

    return E_she_target, alpha, alpha_err, delta_oh, delta_oh_err


def _fmt_tick_sd(val, pos):
    s = f'{val:.2f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


# --- make_second_derivatives_figures (3 PNGs, one per KOH) ---------------------
#
# 6 catalyst rows x 4 CO% cols per KOH figure.
# Each panel: alpha (filled circles, +/-1 SE), delta_OH (open diamonds),
# linear fits in window (solid / dashed), slope annotation top-right.
# highlight=True shades the kinetic window orange and shows all 20 points.

def make_second_derivatives_figures(highlight=False, ylim=True,
                                     save_dir=None, dpi=1200):
    if save_dir is None:
        save_dir = batch_root

    fname_pat = ('second_derivatives_{tag}_M_KOH_highlight.png' if highlight
                 else 'second_derivatives_{tag}_M_KOH.png')
    tags = {250: '0p25', 500: '0p5', 1000: '1'}

    saved = []

    for koh_mM in KOH_KEYS_ASC:
        cache = {}
        for ri, cat in enumerate(catalysts):
            cache[cat] = {}
            for suffix, _ in PRESSURES_SD:
                E, alpha, alpha_err, delta_oh, delta_oh_err = \
                    _load_alpha_oh_for_panel(cat, ri, suffix, koh_mM)
                e20 = compute_e20pct_she(cat, suffix, koh_mM)
                cache[cat][suffix] = dict(
                    E=E, alpha=alpha, alpha_err=alpha_err,
                    delta_oh=delta_oh, delta_oh_err=delta_oh_err, e20=e20)

        # y-limits: True -> autofit window data, (lo, hi) -> explicit.
        panel_ylim = None
        if ylim is True:
            vals = []
            for cat in catalysts:
                for suffix, _ in PRESSURES_SD:
                    d = cache[cat][suffix]
                    e20 = d['e20']
                    win = ((d['E'] <= e20) if not np.isnan(e20)
                           else np.ones(len(d['E']), bool))
                    for arr in (d['alpha'], d['delta_oh']):
                        sub = arr[win]
                        vals.extend(sub[np.isfinite(sub)].tolist())
            if vals:
                ymin, ymax = min(vals), max(vals)
                pad = 0.05 * (ymax - ymin) if ymax != ymin else 0.1
                panel_ylim = (ymin - pad, ymax + pad)
        elif isinstance(ylim, (tuple, list)) and len(ylim) == 2:
            panel_ylim = tuple(ylim)

        fig, axes = plt.subplots(6, 4, figsize=(14, 16),
                                 sharex=False, sharey=False)
        fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.98,
                            hspace=0.25, wspace=0.25)
        fig.suptitle(KOH_TITLES_SD[koh_mM], fontsize=16, fontweight='bold',
                     y=0.97)

        for ri, cat in enumerate(catalysts):
            color   = cat_colors[ri]
            cat_lbl = cat_names[ri]

            for ci, (suffix, co_disp) in enumerate(PRESSURES_SD):
                ax = axes[ri, ci]
                d  = cache[cat][suffix]
                E, alpha, alpha_err = d['E'], d['alpha'], d['alpha_err']
                delta_oh, delta_oh_err = d['delta_oh'], d['delta_oh_err']
                e20 = d['e20']

                if highlight:
                    win = ((E <= e20) if not np.isnan(e20)
                           else np.ones(len(E), bool))
                    ma = np.isfinite(alpha)
                    mo = np.isfinite(delta_oh)
                    if ma.any():
                        ax.errorbar(E[ma], alpha[ma], yerr=alpha_err[ma],
                                    ls='none', marker='o', ms=4,
                                    mfc=color, mec=color, mew=0.4,
                                    ecolor=color, elinewidth=0.8, capsize=3,
                                    alpha=0.30, zorder=2)
                    if mo.any():
                        ax.errorbar(E[mo], delta_oh[mo], yerr=delta_oh_err[mo],
                                    ls='none', marker='D', ms=4,
                                    mfc='none', mec=color, mew=0.6,
                                    ecolor=color, elinewidth=0.8, capsize=3,
                                    alpha=0.30, zorder=2)
                    maw = ma & win
                    mow = mo & win
                    if maw.any():
                        ax.errorbar(E[maw], alpha[maw], yerr=alpha_err[maw],
                                    ls='none', marker='o', ms=5,
                                    mfc=color, mec=color, mew=0.5,
                                    ecolor=color, elinewidth=0.9, capsize=3,
                                    alpha=0.90, zorder=3)
                    if mow.any():
                        ax.errorbar(E[mow], delta_oh[mow], yerr=delta_oh_err[mow],
                                    ls='none', marker='D', ms=5,
                                    mfc='none', mec=color, mew=0.8,
                                    ecolor=color, elinewidth=0.9, capsize=3,
                                    alpha=0.90, zorder=3)
                    if not np.isnan(e20):
                        x_left = float(E.min())
                        ax.axvspan(x_left, e20, alpha=0.14,
                                   color=WINDOW_COLOR, linewidth=0, zorder=1)
                        ax.axvline(e20, color=WINDOW_COLOR, lw=1.2, ls='--',
                                   alpha=0.85, zorder=4)
                else:
                    win = ((E <= e20) if not np.isnan(e20)
                           else np.ones(len(E), bool))
                    ma = win & np.isfinite(alpha)
                    mo = win & np.isfinite(delta_oh)
                    if ma.any():
                        ax.errorbar(E[ma], alpha[ma], yerr=alpha_err[ma],
                                    ls='none', marker='o', ms=5,
                                    mfc=color, mec=color, mew=0.5,
                                    ecolor=color, elinewidth=0.9, capsize=3,
                                    alpha=0.90, zorder=2)
                    if mo.any():
                        ax.errorbar(E[mo], delta_oh[mo], yerr=delta_oh_err[mo],
                                    ls='none', marker='D', ms=5,
                                    mfc='none', mec=color, mew=0.8,
                                    ecolor=color, elinewidth=0.9, capsize=3,
                                    alpha=0.90, zorder=2)

                slope_a_str, slope_o_str = '—', '—'
                if not np.isnan(e20):
                    res_a = linreg_window_sd(E, alpha, e20, robust=True)
                    res_o = linreg_window_sd(E, delta_oh, e20)
                    if res_a is not None:
                        m_a, _, E_w, yf_a = res_a
                        ax.plot(E_w, yf_a, color=color, lw=1.8, ls='-',
                                zorder=5, alpha=0.95)
                        slope_a_str = f'{m_a:.2f}'
                    if res_o is not None:
                        m_o, _, E_w, yf_o = res_o
                        ax.plot(E_w, yf_o, color=color, lw=1.8, ls='--',
                                zorder=5, alpha=0.95)
                        slope_o_str = f'{m_o:.2f}'

                ax.text(0.97, 0.97,
                        f'd$\\alpha$/dE={slope_a_str}\nd$\\delta$/dE={slope_o_str}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color=color,
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  alpha=0.65, lw=0))

                # X-axis: leave auto-scaled per panel (matches source).
                # Y-axis: panel_ylim if requested, else default.
                if panel_ylim is not None:
                    ax.set_ylim(panel_ylim)

                ax.tick_params(axis='both', which='major', labelsize=11,
                               width=0.8, labelcolor='black')
                ax.xaxis.set_major_formatter(_FuncFormatter(_fmt_tick_sd))
                ax.yaxis.set_major_formatter(_FuncFormatter(_fmt_tick_sd))
                if ri == 5:
                    ax.set_xlabel(r'Potential ($V_{SHE}$)',
                                  fontweight='bold', fontsize=12)
                if ci == 0:
                    ax.text(0.05, 0.05, cat_lbl, transform=ax.transAxes,
                            ha='left', va='bottom', fontsize=11,
                            fontweight='bold', color=color)
                if ri == 0:
                    ax.set_title(co_disp, fontweight='bold', fontsize=12, pad=4)

                if ri == 0 and ci == 0:
                    ax.legend(handles=[
                        Line2D([0], [0], color='0.3', lw=0, marker='o', ms=5,
                               mfc='0.3', mec='0.3', label=r'$\alpha$'),
                        Line2D([0], [0], color='0.3', lw=0, marker='D', ms=5,
                               mfc='none', mec='0.3', mew=0.8,
                               label=r'$\delta_{OH^-}$'),
                    ], fontsize=8, framealpha=0.8, loc='upper left')

        out = os.path.join(save_dir, fname_pat.format(tag=tags[koh_mM]))
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        saved.append(out)
        print(f'Saved: {os.path.basename(out)}')

    return saved


make_second_derivatives_figures(highlight=True, ylim=(-0.5, 1.5))
