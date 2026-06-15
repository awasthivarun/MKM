# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:10:56 2023

@author: twhit
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

for _k, _v in {
    'font.sans-serif':         'Arial',
    'font.weight':             'bold',
    'font.size':               12,
    'lines.linewidth':         2,
    'lines.linestyle':         '-',
    'axes.labelsize':          14,
    'axes.labelweight':        'bold',
    'lines.marker':            'o',
    'lines.markeredgecolor':   'black',
    'lines.markeredgewidth':   '2',
    'lines.markersize':        10,
    'mathtext.default':        'regular',
    'errorbar.capsize':        3,
    'legend.frameon':          False,
    'legend.handletextpad':    0.05,
}.items():
    mpl.rcParams[_k] = _v

# Self-locate (cross-platform: every path is built with os.path.join, so
# Windows / macOS / Linux all work without separator assumptions).
# __file__ is defined when run as a script; if this code is pasted into a
# Jupyter cell where __file__ is undefined, fall back to cwd (Jupyter sets
# cwd to the notebook's folder when opened from the file browser).
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# material = 'Pd100'
# material = 'Ag10Pd90'
# material = 'Ag25Pd75'
material = 'Ag50Pd50'
# material = 'Ag75Pd25'
# material = 'Ag90Pd10'

# model = "single_site_LH_CO_differential"
# model = "single_site_ER_CO_differential"
# model = "dual_site_BLH_CO_differential"
model = "dual_site_BLH_PCT_CO_differential"
# model = "single_site_LH_CO_differential_with_lat_int"
# model = "single_site_ER_CO_differential_with_lat_int"
# model = "dual_site_BLH_CO_differential_with_lat_int"
# model = "dual_site_BLH_PCT_CO_differential_with_lat_int"

df1 = pd.read_excel(os.path.join(SCRIPT_DIR, material, f'{material}_{model}_coverages_rates.xlsx'))
df2 = pd.read_excel(os.path.join(SCRIPT_DIR, material, f'{material}_{model}_observables.xlsx'))

if model[0:6] == "single":
        fig, axs = plt.subplots(2,2,figsize=(10,6))
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 0.1% CO, 0.5 M KOH - CO order'], "k-", linewidth=1, label="CO*")
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 1% CO, 0.5 M KOH - CO order'], "k--", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 10% CO, 0.5 M KOH - CO order'], "k:", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 100% CO, 0.5 M KOH - CO order'], "k-.", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH at 0.1% CO, 0.5 M KOH - CO order'], "c-", linewidth=1, label="OH*")
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH at 1% CO, 0.5 M KOH - CO order'], "c--", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH at 10% CO, 0.5 M KOH - CO order'], "c:", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH at 100% CO, 0.5 M KOH - CO order'], "c-.", linewidth=1)
        axs[0,0].set_ylabel("Coverage", fontsize=18)
        axs[0,0].tick_params(labelsize=16)
        axs[0,0].tick_params(axis="x",labelbottom = False, width=1.5)
        axs[0,0].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[0,0].set_xlim([0.7,1.0])
        elif material == "Ag10Pd90":
            axs[0,0].set_xlim([0.7,1.0])
        else:
            axs[0,0].set_xlim([0.55,1.0])
            axs[0,0].set_ylim([0,1])
            axs[0,0].legend(fontsize=16)
        
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 0.25 M KOH - OH order'], "k-", linewidth=1, label="CO*")
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 0.5 M KOH - OH order'], "k--", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 1 M KOH - OH order'], "k:", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH at 10% CO, 0.25 M KOH - OH order'], "c-", linewidth=1, label="OH*")
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH at 10% CO, 0.5 M KOH - OH order'], "c--", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH at 10% CO, 1 M KOH - OH order'], "c:", linewidth=1)
        axs[0,1].tick_params(axis="x",labelbottom = False, width=1.5)
        axs[0,1].tick_params(labelsize=16)
        axs[0,1].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[0,1].set_xlim([-0.1,0.2])
        elif material == "Ag10Pd90":
            axs[0,1].set_xlim([-0.1,0.2])
        else:
            axs[0,1].set_xlim([-0.25,0.2])
        axs[0,1].set_ylim([0,1])
        axs[0,1].legend(fontsize=16)
        
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 0.1% CO, 0.5 M KOH - CO order'], "k-", linewidth=1, label="0.1% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 1% CO, 0.5 M KOH - CO order'], "k--", linewidth=1, label="1% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 10% CO, 0.5 M KOH - CO order'], "k:", linewidth=1, label="10% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 100% CO, 0.5 M KOH - CO order'], "k-.", linewidth=1, label="100% CO")
        axs[1,0].set_xlabel("Potential (V vs. RHE)", fontsize=18)
        axs[1,0].set_ylabel("Rate (s$^{-1}$)", fontsize=18)
        axs[1,0].tick_params(labelsize=16)
        axs[1,0].tick_params(axis="x", width=1.5,labelsize=16)
        axs[1,0].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[1,0].set_xlim([0.7,1.0])
        elif material == "Ag10Pd90":
            axs[1,0].set_xlim([0.7,1.0])
        else:
            axs[1,0].set_xlim([0.55,1.0])
        axs[1,0].legend(fontsize=16)
        
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 0.25 M KOH - OH order'], "k-", linewidth=1, label="0.25 M")
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 0.5 M KOH - OH order'], "k--", linewidth=1, label="0.5 M")
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 1 M KOH - OH order'], "k:", linewidth=1, label="1 M")
        axs[1,1].set_xlabel("Potential (V vs. SHE)", fontsize=18)
        # axs[1,1].set_ylabel("Rate (s$^{-1}$)", fontsize=14)
        axs[1,1].tick_params(labelsize=16)
        axs[1,1].tick_params(axis="x", width=1.5, labelsize=16)
        axs[1,1].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[1,1].set_xlim([-0.1,0.2])
        elif material == "Ag10Pd90":
            axs[1,1].set_xlim([-0.1,0.2])
        else:
            axs[1,1].set_xlim([-0.25,0.2])
        axs[1,1].legend(fontsize=16)

        plt.tight_layout(pad=0.5)
        fig.savefig(os.path.join(SCRIPT_DIR, f'{material}_{model}_coverages_rates.png'),
                dpi=1200, bbox_inches='tight')
else:
        fig, axs = plt.subplots(2,2,figsize=(10,6))
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 0.1% CO, 0.5 M KOH - CO order'], "k-", linewidth=1, label="CO*")
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 1% CO, 0.5 M KOH - CO order'], "k--", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 10% CO, 0.5 M KOH - CO order'], "k:", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_CO at 100% CO, 0.5 M KOH - CO order'], "k-.", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S1 at 0.1% CO, 0.5 M KOH - CO order'], "c-", linewidth=1, label="OH*")
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S1 at 1% CO, 0.5 M KOH - CO order'], "c--", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S1 at 10% CO, 0.5 M KOH - CO order'], "c:", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S1 at 100% CO, 0.5 M KOH - CO order'], "c-.", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S2 at 0.1% CO, 0.5 M KOH - CO order'], "r-", linewidth=1, label="OH$^\#$")
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S2 at 1% CO, 0.5 M KOH - CO order'], "r--", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S2 at 10% CO, 0.5 M KOH - CO order'], "r:", linewidth=1)
        axs[0,0].plot(df1['Model potential (V vs RHE)'], df1['Theta_OH_S2 at 100% CO, 0.5 M KOH - CO order'], "r-.", linewidth=1)
        axs[0,0].set_ylabel("Coverage", fontsize=18)
        axs[0,0].tick_params(axis="x",labelbottom = False, width=1.5)
        axs[0,0].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[0,0].set_xlim([0.7,1.0])
        elif material == "Ag10Pd90":
            axs[0,0].set_xlim([0.7,1.0])
        else:
            axs[0,0].set_xlim([0.55,1.0])
        axs[0,0].set_ylim([0,1])
        axs[0,0].legend(fontsize=16)
        
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 0.25 M KOH - OH order'], "k-", linewidth=1, label="CO*")
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 0.5 M KOH - OH order'], "k--", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_CO at 10% CO, 1 M KOH - OH order'], "k:", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S1 at 10% CO, 0.25 M KOH - OH order'], "c-", linewidth=1, label="OH*")
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S1 at 10% CO, 0.5 M KOH - OH order'], "c--", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S1 at 10% CO, 1 M KOH - OH order'], "c:", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S2 at 10% CO, 0.25 M KOH - OH order'], "r-", linewidth=1, label="OH$^\#$")
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S2 at 10% CO, 0.5 M KOH - OH order'], "r--", linewidth=1)
        axs[0,1].plot(df1['Model potential (V vs SHE)'], df1['Theta_OH_S2 at 10% CO, 1 M KOH - OH order'], "r:", linewidth=1)
        axs[0,1].tick_params(axis="x",labelbottom = False, width=1.5)
        axs[0,1].tick_params(axis="y",labelsize = 16, width=1.5)
        if material == "Pd100":
            axs[0,1].set_xlim([-0.1,0.2])
        elif material == "Ag10Pd90":
            axs[0,1].set_xlim([-0.1,0.2])
        else:
            axs[0,1].set_xlim([-0.25,0.2])
        axs[0,1].set_ylim([0,1])
        axs[0,1].legend(fontsize=16)
        
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 0.1% CO, 0.5 M KOH - CO order'], "k-", linewidth=1, label="0.1% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 1% CO, 0.5 M KOH - CO order'], "k--", linewidth=1, label="1% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 10% CO, 0.5 M KOH - CO order'], "k:", linewidth=1, label="10% CO")
        axs[1,0].plot(df1['Model potential (V vs RHE)'], df1['Total Rate at 100% CO, 0.5 M KOH - CO order'], "k-.", linewidth=1, label="100% CO")
        axs[1,0].set_xlabel("Potential (V vs. RHE)", fontsize=18)
        axs[1,0].set_ylabel("Rate (s$^{-1}$)", fontsize=18)
        axs[1,0].tick_params(axis="x", width=1.5, labelsize=16)
        axs[1,0].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[1,0].set_xlim([0.7,1.0])
        elif material == "Ag10Pd90":
            axs[1,0].set_xlim([0.7,1.0])
        else:
            axs[1,0].set_xlim([0.55,1.0])
        axs[1,0].legend(fontsize=16)
        
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 0.25 M KOH - OH order'], "k-", linewidth=1, label="0.25 M")
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 0.5 M KOH - OH order'], "k--", linewidth=1, label="0.5 M")
        axs[1,1].plot(df1['Model potential (V vs SHE)'], df1['Total Rate at 10% CO, 1 M KOH - OH order'], "k:", linewidth=1, label="1 M")
        axs[1,1].set_xlabel("Potential (V vs. SHE)", fontsize=18)
        # axs[1,1].set_ylabel("Rate (s$^{-1}$)", fontsize=14)
        axs[1,1].tick_params(axis="x", width=1.5, labelsize=16)
        axs[1,1].tick_params(axis="y",labelsize=16, width=1.5)
        if material == "Pd100":
            axs[1,1].set_xlim([-0.1,0.2])
        elif material == "Ag10Pd90":
            axs[1,1].set_xlim([-0.1,0.2])
        else:
            axs[1,1].set_xlim([-0.25,0.2])
        axs[1,1].legend(fontsize=16)

        plt.tight_layout(pad=0.5)
        fig.savefig(os.path.join(SCRIPT_DIR, f'{material}_{model}_coverages_rates.png'),
                dpi=1200, bbox_inches='tight')

fig, axs = plt.subplots(1,3,figsize=(15,4.5))

if material == 'Pd100':
    plot_color='k'
elif material == 'Ag10Pd90':
    plot_color=(0.4,0.0,0.4)
elif material == 'Ag25Pd75':
    plot_color=(0.7,0.0,0.7)
elif material == 'Ag50Pd50':
    plot_color=(0.0,0.0,0.6)
elif material=='Ag75Pd25':
    plot_color=(0.0,0.0,1.0)
else:
    plot_color=(0.0,0.5,1.0)

# Individual fit: apparent transfer coefficient on experimental data
axs[0].plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 0.1% CO, 0.5 M KOH'], color=plot_color, linestyle='-.', linewidth = 1, label='0.1% CO', marker='none')
axs[0].plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 1% CO, 0.5 M KOH'], color=plot_color, linestyle=':', linewidth = 1, label='1% CO', marker='none')
axs[0].plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 10% CO, 0.5 M KOH'], color=plot_color, linestyle='--', linewidth = 1, label='10% CO', marker='none')
axs[0].plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 100% CO, 0.5 M KOH'], color=plot_color, linestyle='-', linewidth = 1, label='100% CO', marker='none')
axs[0].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH'], 
                  yerr = df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                  alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'],
               marker='_', color=plot_color, alpha=0.333)
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.333)
axs[0].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH'], 
                  yerr = df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                  alpha = 0.5, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.5)
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.5)
axs[0].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH'], 
                  yerr = df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                  alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.666)
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.666)
axs[0].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH'], 
                  yerr = df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                  alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=0.666)
axs[0].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], 
               marker='_', color=plot_color, alpha=1)
axs[0].set_xlabel("Potential ($V_{RHE}$)", fontsize=28)
axs[0].set_ylabel("$\\alpha$", fontsize=28)
axs[0].tick_params(axis="x", width=1.5, labelsize=22)
axs[0].tick_params(axis="y",labelsize=22, width=1.5)
if material == "Pd100":
    axs[0].set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
    axs[0].set_xlim([0.7,1.0])
elif material == "Ag10Pd90":
    axs[0].set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
    axs[0].set_xlim([0.7,1.0])
else:
    axs[0].set_xticks([0.6,0.8,1],['0.6','0.8','1'])
    axs[0].set_xlim([0.55,1])
# axs[0].set_yticks([-0.5,0,0.5,1],['-0.5','0','0.5','1'])
axs[0].set_yticks([-0.5,-0.25,0,0.25,0.5],['-0.5','-0.25','0','0.25','0.5'])
axs[0].set_ylim([-0.55,0.55])
axs[0].legend(ncol=1, loc="lower left", columnspacing=0.25, fontsize=16)

# # Individual fit: apparent CO reaction order on experimental data
axs[1].plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 0.1% - 1% CO'], color=plot_color, linestyle=':', linewidth = 1, label='0.1-1%', marker='none')
axs[1].plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 1% - 10% CO'], color=plot_color, linestyle='--', linewidth = 1, label='1-10%', marker='none')
axs[1].plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 10% - 100% CO'], color=plot_color, linestyle='-', linewidth = 1, label='10-100%', marker='none')
axs[1].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO'], 
                  yerr = df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], color=plot_color, fmt = "o", 
                  alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO']+df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], 
               marker='_', color=plot_color, alpha=0.333)
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO']-df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], 
               marker='_', color=plot_color, alpha=0.333)
axs[1].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO'], 
                  yerr = df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], color=plot_color, fmt = "o", 
                  alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO']+df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], 
               marker='_', color=plot_color, alpha=0.666)
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO']-df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], 
               marker='_', color=plot_color, alpha=0.666)
axs[1].errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO'], 
                  yerr = df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], color=plot_color, fmt = "o", 
                  alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO']+df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], 
               marker='_', color=plot_color, alpha=1)
axs[1].scatter(df2['Experimental potential (V vs RHE)'], 
               df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO']-df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], 
               marker='_', color=plot_color, alpha=1)
axs[1].set_xlabel("Potential ($V_{RHE}$)", fontsize=28)
axs[1].set_ylabel("$\\delta_{CO}$", fontsize=28)
axs[1].tick_params(axis="x",labelsize=22, width=1.5)
axs[1].tick_params(axis="y",labelsize=22, width=1.5)
if material == "Pd100":
    axs[1].set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
    axs[1].set_xlim([0.7,1.0])
elif material == "Ag10Pd90":
    axs[1].set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
    axs[1].set_xlim([0.7,1.0])
else:
    axs[1].set_xticks([0.6,0.8,1],['0.6','0.8','1'])
    axs[1].set_xlim([0.55,1])
axs[1].set_yticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'])
axs[1].set_ylim([-0.05,1.05])
axs[1].legend(ncol=1, loc="upper left", columnspacing=0.25, fontsize=16)

# Individual fit: apparent OH reaction order on experimental data
# axs[2].plot(df2['Model potential (V vs RHE)']-0.8, df2['Model OH order at 0.1% CO'], color=plot_color, linestyle='-.', linewidth = 1, label='0.1% CO', marker='none')
# axs[2].plot(df2['Model potential (V vs RHE)']-0.8, df2['Model OH order at 1% CO'], color=plot_color, linestyle=':', linewidth = 1, label='1% CO', marker='none')
# axs[2].plot(df2['Model potential (V vs RHE)']-0.8, df2['Model OH order at 10% CO'], color=plot_color, linestyle='--', linewidth = 1, label='10% CO', marker='none')
# axs[2].plot(df2['Model potential (V vs RHE)']-0.8, df2['Model OH order at 100% CO'], color=plot_color, linestyle='-', linewidth = 1, label='100% CO', marker='none')
axs[2].plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 0.1% CO'], color=plot_color, linestyle='-.', linewidth = 1, label='0.1% CO', marker='none')
axs[2].plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 1% CO'], color=plot_color, linestyle=':', linewidth = 1, label='1% CO', marker='none')
axs[2].plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 10% CO'], color=plot_color, linestyle='--', linewidth = 1, label='10% CO', marker='none')
axs[2].plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 100% CO'], color=plot_color, linestyle='-', linewidth = 1, label='100% CO', marker='none')
axs[2].errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 0.1% CO'], 
                  yerr = df2['Experimental OH order error at 0.1% CO'], color=plot_color, fmt = "o", 
                  alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 0.1% CO']+df2['Experimental OH order error at 0.1% CO'], 
               marker='_', color=plot_color, alpha=0.333)
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 0.1% CO']-df2['Experimental OH order error at 0.1% CO'], 
               marker='_', color=plot_color, alpha=0.333)
axs[2].errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 1% CO'], 
                  yerr = df2['Experimental OH order error at 1% CO'], color=plot_color, fmt = "o", 
                  alpha = 0.5, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 1% CO']+df2['Experimental OH order error at 1% CO'], 
               marker='_', color=plot_color, alpha=0.5)
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 1% CO']-df2['Experimental OH order error at 1% CO'], 
               marker='_', color=plot_color, alpha=0.5)
axs[2].errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 10% CO'], 
                  yerr = df2['Experimental OH order error at 10% CO'], color=plot_color, fmt = "o", 
                  alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 10% CO']+df2['Experimental OH order error at 10% CO'], 
               marker='_', color=plot_color, alpha=0.666)
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 10% CO']-df2['Experimental OH order error at 10% CO'], 
               marker='_', color=plot_color, alpha=0.666)
axs[2].errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 100% CO'], 
                  yerr = df2['Experimental OH order error at 100% CO'], color=plot_color, fmt = "o", 
                  alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                  ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 100% CO']+df2['Experimental OH order error at 100% CO'], 
               marker='_', color=plot_color, alpha=1)
axs[2].scatter(df2['Experimental potential (V vs SHE)'], 
               df2['Experimental OH order at 100% CO']-df2['Experimental OH order error at 100% CO'], 
               marker='_', color=plot_color, alpha=1)
axs[2].set_xlabel("Potential ($V_{SHE}$)", fontsize=28)
axs[2].set_ylabel("$\\delta_{OH^-}$", fontsize=28)
axs[2].tick_params(axis="x",labelsize=22, width=1.5)
axs[2].tick_params(axis="y",labelsize=22, width=1.5)
if material == "Pd100":
    axs[2].set_xticks([-0.05,0.05,0.15],['-0.05','0.05','0.15'])
    axs[2].set_xlim([-0.1,0.2])
elif material == "Ag10Pd90":
    axs[2].set_xticks([-0.05,0.05,0.15],['-0.05','0.05','0.15'])
    axs[2].set_xlim([-0.1,0.2])
else:
    axs[2].set_xticks([-0.2,0,0.2],['-0.2','0','0.2'])
    axs[2].set_xlim([-0.25,0.2])
axs[2].set_yticks([-0.5,0,0.5,1,1.5],['-0.5','0','0.5','1','1.5'])
axs[2].set_ylim([-0.55,1.55])
axs[2].legend(ncol=1, loc="lower left", columnspacing=0.25, fontsize=16)

plt.tight_layout(pad=0.5)
fig.savefig(os.path.join(SCRIPT_DIR, f'{material}_{model}_observables.png'),
                dpi=1200, bbox_inches='tight')

if material == "Ag50Pd50" and model == "dual_site_BLH_PCT_CO_differential":
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:6])
    # ── ax1: experimental diff data from CSV (0.5 M KOH, all CO pressures) ──────
    _slopes_csv = os.path.join(SCRIPT_DIR, 'second_derivatives.csv')

    # Always load the original CSV so _df_05 is available to later figures too
    _df_csv = pd.read_csv(_slopes_csv)
    _df_05  = _df_csv[_df_csv['KOH_M'] == 0.5].copy()

    _cat_colors = {
        'Pd':   'k',
        'Ag10': (0.4, 0.0, 0.4),
        'Ag25': (0.7, 0.0, 0.7),
        'Ag50': (0.0, 0.0, 0.6),
        'Ag75': (0.0, 0.0, 1.0),
        'Ag90': (0.0, 0.5, 1.0),
    }
    # Full-name → short-name mapping (for slopes CSV → color lookup)
    _cat_short = {
        'Pd100': 'Pd', 'Ag10Pd90': 'Ag10', 'Ag25Pd75': 'Ag25',
        'Ag50Pd50': 'Ag50', 'Ag75Pd25': 'Ag75', 'Ag90Pd10': 'Ag90',
    }
    _df_sl = pd.read_csv(_slopes_csv)
    _df_sl05 = _df_sl[_df_sl['KOH_M'] == 0.5].copy()
    for _cat_full, _grp in _df_sl05.groupby('catalyst'):
        _short = _cat_short.get(_cat_full, _cat_full)
        _color = _cat_colors.get(_short, '0.5')
        _ag = _grp['Ag_content_at%'].iloc[0]
        _diffs = _grp['diff_slope'].dropna().values
        if len(_diffs) == 0:
            continue
        _ymean = float(_diffs.mean())
        _yerr  = float(_diffs.std(ddof=0))
        ax1.scatter(_ag, _ymean, color=_color, edgecolor='k',
                    marker='o', s=250, zorder=3)
        ax1.scatter(_ag, _ymean + _yerr, color=_color, marker='_',
                    s=120, zorder=3)
        ax1.scatter(_ag, _ymean - _yerr, color=_color, marker='_',
                    s=120, zorder=3)
        ax1.errorbar(_ag, _ymean, _yerr, ecolor=_color, marker='none',
                     elinewidth=3, capthick=0, zorder=2)

    #ax1.axhline(0, color='0.6', linewidth=0.8, linestyle='--', zorder=1)
    ax1.set_xlabel('Ag composition (%)', fontsize=28)
    ax1.set_ylabel('$\\frac{d\\delta_{OH^-}}{dE}$ - $\\frac{d\\alpha}{dE}$', fontsize=28)
    ax1.set_xlim(-10, 110)
    ax1.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100'])
    ax1.set_ylim(-3, 1.5)
    ax1.set_yticks([-3, -2, -1, 0, 1], ['-3', '-2', '-1', '0', '1'])
    ax1.tick_params(labelsize=22)
    ax2.scatter([0,25,50,75],[-0.747293833, -0.732170833, -0.682419833, -0.656479167], marker='s', color='k', linewidth=2, s=300, edgecolor='k')
    ax2.scatter([0,25,50,75], [x + y for x, y in zip([-0.747293833, -0.732170833, -0.682419833, -0.656479167], [0.02, 0.015, 0.05, 0.06])], marker='_', s=100, color='k')
    ax2.scatter([0,25,50,75], [x - y for x, y in zip([-0.747293833, -0.732170833, -0.682419833, -0.656479167], [0.02, 0.015, 0.05, 0.06])], marker='_', s=100, color='k')
    ax2.errorbar([0,25,50,75], [-0.747293833, -0.732170833, -0.682419833, -0.656479167], [0.02, 0.015, 0.05, 0.06], ecolor='k', marker='none', elinewidth=3, capthick=0, linewidth=0)
    ax2.scatter([25,50,75,100],[-0.5054455, -0.483812167, -0.483449333, -0.482583167], marker='^', color='limegreen', linewidth=2, s=300, edgecolor='k')
    ax2.scatter([25,50,75,100], [x + y for x, y in zip([-0.5054455, -0.483812167, -0.483449333, -0.482583167], [0.02, 0.02, 0.025, 0.04])], marker='_', s=100, color='limegreen')
    ax2.scatter([25,50,75,100], [x - y for x, y in zip([-0.5054455, -0.483812167, -0.483449333, -0.482583167], [0.02, 0.02, 0.025, 0.04])], marker='_', s=100, color='limegreen')
    ax2.errorbar([25,50,75,100], [-0.5054455, -0.483812167, -0.483449333, -0.482583167], [0.02, 0.02, 0.025, 0.04], ecolor='limegreen', marker='none', elinewidth=3, capthick=0, linewidth=0)
    ax2.set_xlim(-10,110)
    ax2.set_xticks([0,25,50,75,100],['0','25','50','75','100'])
    ax2.set_ylim(-1,0)
    ax2.set_yticks([-1,-0.75,-0.5,-0.25,0],['-1','-0.75','-0.5','-0.25','0'])
    ax2.set_xlabel('Ag composition (%)', fontsize=28)
    ax2.tick_params(labelsize=22)
    ax2.set_ylabel('$\mu_{calc}$', fontsize=28)
    ax2.text(10, -0.9, '$OH^*$', fontsize=28, ha='center', color='k')
    ax2.text(60, -0.4, '$OH^\#$', fontsize=28, ha='center', color='limegreen')
    ax3.plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 0.1% CO, 0.5 M KOH'], color=plot_color, linestyle='-.', linewidth = 1, label='0.1% CO', marker='none')
    ax3.plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 1% CO, 0.5 M KOH'], color=plot_color, linestyle=':', linewidth = 1, label='1% CO', marker='none')
    ax3.plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 10% CO, 0.5 M KOH'], color=plot_color, linestyle='--', linewidth = 1, label='10% CO', marker='none')
    ax3.plot(df2['Model potential (V vs RHE)'], df2['Model transfer coefficient at 100% CO, 0.5 M KOH'], color=plot_color, linestyle='-', linewidth = 1, label='100% CO', marker='none')
    ax3.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH'], 
                      yerr = df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                      alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'],
                   marker='_', color=plot_color, alpha=0.333)
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 0.1% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.333)
    ax3.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH'], 
                      yerr = df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                      alpha = 0.5, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.5)
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 1% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 1% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.5)
    ax3.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH'], 
                      yerr = df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                      alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 10% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 10% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax3.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH'], 
                      yerr = df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], color=plot_color, fmt = "o", 
                      alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH']+df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax3.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental transfer coefficient at 100% CO, 0.5 M KOH']-df2['Experimental transfer coefficient error at 100% CO, 0.5 M KOH'], 
                   marker='_', color=plot_color, alpha=1)
    ax3.set_xlabel("Potential ($V_{RHE}$)", fontsize=28)
    ax3.set_ylabel("$\\alpha$", fontsize=28)
    ax3.tick_params(axis="x", width=1.5, labelsize=22)
    ax3.tick_params(axis="y",labelsize=22, width=1.5)
    if material == "Pd100":
        ax3.set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
        ax3.set_xlim([0.7,1.0])
    elif material == "Ag10Pd90":
        ax3.set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
        ax3.set_xlim([0.7,1.0])
    else:
        ax3.set_xticks([0.6,0.8,1],['0.6','0.8','1'])
        ax3.set_xlim([0.55,1])
    # ax3.set_yticks([-0.5,0,0.5,1],['-0.5','0','0.5','1'])
    ax3.set_yticks([-0.5,-0.25,0,0.25,0.5],['-0.5','-0.25','0','0.25','0.5'])
    ax3.set_ylim([-0.55,0.55])
    ax3.legend(ncol=1, loc="lower left", columnspacing=0.25, fontsize=16)
    ax4.plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 0.1% - 1% CO'], color=plot_color, linestyle=':', linewidth = 1, label='0.1-1%', marker='none')
    ax4.plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 1% - 10% CO'], color=plot_color, linestyle='--', linewidth = 1, label='1-10%', marker='none')
    ax4.plot(df2['Model potential (V vs RHE)'], df2['Model CO order at 0.5 M KOH, 10% - 100% CO'], color=plot_color, linestyle='-', linewidth = 1, label='10-100%', marker='none')
    ax4.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO'], 
                      yerr = df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], color=plot_color, fmt = "o", 
                      alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO']+df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], 
                   marker='_', color=plot_color, alpha=0.333)
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 0.1% - 1% CO']-df2['Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO'], 
                   marker='_', color=plot_color, alpha=0.333)
    ax4.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO'], 
                      yerr = df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], color=plot_color, fmt = "o", 
                      alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO']+df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 1% - 10% CO']-df2['Experimental CO order error at 0.5 M KOH, 1% - 10% CO'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax4.errorbar(df2['Experimental potential (V vs RHE)'], df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO'], 
                      yerr = df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], color=plot_color, fmt = "o", 
                      alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO']+df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], 
                   marker='_', color=plot_color, alpha=1)
    ax4.scatter(df2['Experimental potential (V vs RHE)'], 
                   df2['Experimental CO order at 0.5 M KOH, 10% - 100% CO']-df2['Experimental CO order error at 0.5 M KOH, 10% - 100% CO'], 
                   marker='_', color=plot_color, alpha=1)
    ax4.set_xlabel("Potential ($V_{RHE}$)", fontsize=28)
    ax4.set_ylabel("$\\delta_{CO}$", fontsize=28)
    ax4.tick_params(axis="x",labelsize=22, width=1.5)
    ax4.tick_params(axis="y",labelsize=22, width=1.5)
    if material == "Pd100":
        ax4.set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
        ax4.set_xlim([0.7,1.0])
    elif material == "Ag10Pd90":
        ax4.set_xticks([0.75,0.85,0.95],['0.75','0.85','0.95'])
        ax4.set_xlim([0.7,1.0])
    else:
        ax4.set_xticks([0.6,0.8,1],['0.6','0.8','1'])
        ax4.set_xlim([0.55,1])
    ax4.set_yticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'])
    ax4.set_ylim([-0.05,1.05])
    ax4.legend(ncol=1, loc="upper left", columnspacing=0.25, fontsize=16)
    ax5.plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 0.1% CO'], color=plot_color, linestyle='-.', linewidth = 1, label='0.1% CO', marker='none')
    ax5.plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 1% CO'], color=plot_color, linestyle=':', linewidth = 1, label='1% CO', marker='none')
    ax5.plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 10% CO'], color=plot_color, linestyle='--', linewidth = 1, label='10% CO', marker='none')
    ax5.plot(df2['Model potential (V vs SHE)'], df2['Model OH order at 100% CO'], color=plot_color, linestyle='-', linewidth = 1, label='100% CO', marker='none')
    ax5.errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 0.1% CO'], 
                      yerr = df2['Experimental OH order error at 0.1% CO'], color=plot_color, fmt = "o", 
                      alpha = 0.333, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 0.1% CO']+df2['Experimental OH order error at 0.1% CO'], 
                   marker='_', color=plot_color, alpha=0.333)
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 0.1% CO']-df2['Experimental OH order error at 0.1% CO'], 
                   marker='_', color=plot_color, alpha=0.333)
    ax5.errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 1% CO'], 
                      yerr = df2['Experimental OH order error at 1% CO'], color=plot_color, fmt = "o", 
                      alpha = 0.5, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 1% CO']+df2['Experimental OH order error at 1% CO'], 
                   marker='_', color=plot_color, alpha=0.5)
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 1% CO']-df2['Experimental OH order error at 1% CO'], 
                   marker='_', color=plot_color, alpha=0.5)
    ax5.errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 10% CO'], 
                      yerr = df2['Experimental OH order error at 10% CO'], color=plot_color, fmt = "o", 
                      alpha = 0.666, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 10% CO']+df2['Experimental OH order error at 10% CO'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 10% CO']-df2['Experimental OH order error at 10% CO'], 
                   marker='_', color=plot_color, alpha=0.666)
    ax5.errorbar(df2['Experimental potential (V vs SHE)'], df2['Experimental OH order at 100% CO'], 
                      yerr = df2['Experimental OH order error at 100% CO'], color=plot_color, fmt = "o", 
                      alpha = 1, markersize=10, markeredgewidth = 1, elinewidth=1, 
                      ecolor=plot_color, capsize=0, capthick=0, markeredgecolor="k")
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 100% CO']+df2['Experimental OH order error at 100% CO'], 
                   marker='_', color=plot_color, alpha=1)
    ax5.scatter(df2['Experimental potential (V vs SHE)'], 
                   df2['Experimental OH order at 100% CO']-df2['Experimental OH order error at 100% CO'], 
                   marker='_', color=plot_color, alpha=1)
    ax5.set_xlabel("Potential ($V_{SHE}$)", fontsize=28)
    ax5.set_ylabel("$\\delta_{OH^-}$", fontsize=28)
    ax5.tick_params(axis="x",labelsize=22, width=1.5)
    ax5.tick_params(axis="y",labelsize=22, width=1.5)
    if material == "Pd100":
        ax5.set_xticks([-0.05,0.05,0.15],['-0.05','0.05','0.15'])
        ax5.set_xlim([-0.1,0.2])
    elif material == "Ag10Pd90":
        ax5.set_xticks([-0.05,0.05,0.15],['-0.05','0.05','0.15'])
        ax5.set_xlim([-0.1,0.2])
    else:
        ax5.set_xticks([-0.2,0,0.2],['-0.2','0','0.2'])
        ax5.set_xlim([-0.25,0.2])
    ax5.set_yticks([-0.5,0,0.5,1,1.5],['-0.5','0','0.5','1','1.5'])
    ax5.set_ylim([-0.55,1.55])
    ax5.legend(ncol=1, loc="lower left", columnspacing=0.25, fontsize=16)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(os.path.join(SCRIPT_DIR, f'{material}_{model}_summary.png'),
                dpi=1200, bbox_inches='tight')
