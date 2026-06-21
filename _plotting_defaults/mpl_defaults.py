# _utils/mpl_defaults.py
"""
A helper module to set custom Matplotlib default parameters.
Importing this module applies the rcParams overrides defined below.
You can also reset to Matplotlib defaults via mpl_defaults.reset().
"""

import matplotlib as mpl
import matplotlib.text as mtext 

# Define your custom rcParam overrides here.
_new_rc = {
            'font.family': "sans-serif",
            'font.sans-serif': ['Arial'],
            'font.weight': 'bold',
            'font.size': 12,
            'axes.titleweight': 'bold',
            'axes.labelsize': 14,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'lines.linewidth': 2,
            'lines.markersize': 10,
            #'mathtext.default': 'regular',
            'errorbar.capsize': 3,
            'legend.frameon': False,
            'legend.handletextpad': 0.3,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'legend.borderpad': 0,
            'legend.columnspacing': 1,
            }

mpl.rcParams.update(_new_rc)

mtext.Text.set_text = (lambda f: (lambda self, s: f(self, None if s is None else s.replace('-', '\N{MINUS SIGN}'))))(mtext.Text.set_text)