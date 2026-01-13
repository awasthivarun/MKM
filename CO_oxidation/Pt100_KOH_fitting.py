'''
Processed data is stored in pkl file
Load data anbd fit with PyMC
Pt100 KOH CO oxidation 

Models: 

A) ER: 
1. CO + * -> CO* 
2. CO* + OH- -> COOH* + e- 
3. COOH* + OH- -> CO2 + H2O + * + e-

'''

import numpy as np
import pymc as pm
import arviz as az
import pickle 
import matplotlib.pyplot as plt
import itertools

# Experimental conditions and constants 
C_KOH_list = np.array([0.1, 0.25, 0.5, 1]) # in M 
P_CO_list = 0.01*np.array([0.1, 1, 10, 100]) # in atm 
conditions = list(itertools.product(C_KOH_list, P_CO_list))
R, T, F = 8.3145, 298.15, 96485
kb, h, kb_J = 8.617e-5, 6.626e-34, 1.3806e-23  # eV/K, J.s, J/K 

# Load and parse data
with open('Pt100_KOH.pkl', 'rb') as f:
    experiments_interp = pickle.load(f)
E_obs = experiments_interp[(0.1, 0.1)]['E'] # V vs RHE - random conditions
E_SHE_obs = experiments_interp[(0.1, 0.1)]['E_SHE'] # V vs SHE - random conditions


# Eley-Rideal Model - 3rd step is RDS  
with pm.Model() as ER_3:
    '''
    1. CO + * -> CO* 
    2. CO* + OH- -> COOH* + (e-) 
    3. COOH* + OH- -> CO2 + H2O + * + (e-) (RDS)
    '''
    # Priors
    beta = pm.Uniform('beta', lower=0, upper=1)
    deltaG1_0 = pm.Normal('deltaG1_0', mu=-0.2, sigma=0.1)
    deltaG2_0 = pm.Normal('deltaG2_0', mu=0.0, sigma=0.1)
    Gact3_0 = pm.Normal('Gact3_0', mu=0.6, sigma=0.2)

    for a_OH in C_KOH_list: 
        E = E_obs - 0.0592*(14 + np.log10(a_OH))
        # Thermodynamics 
        deltaG1 = deltaG1_0
        deltaG2 = deltaG2_0 - E
        Gact3 = Gact3_0 - beta*E
        K1 = pm.math.exp(-deltaG1/(kb*T))
        K2 = pm.math.exp(-deltaG2/(kb*T))
        k3 = (kb_J*T/h) * pm.math
        
        for P_CO in P_CO_list: 
            # Equilibriated reactions
            theta = 1 / (1 + K1*P_CO + K2*K1*P_CO*a_OH)
            theta_CO = K1 * P_CO * theta
            theta_COOH = K2 * theta_CO * a_OH

            # Rate expression
            # TODO: Need to be changed to store rate for each condition
            rate = k3 * theta_COOH * a_OH

    # Kinetic observables
    log_rate = pm.math.log(rate)
    mu_alpha = - (R*T/F) * 
    mu_delta_OH = # see experiment -> observables data
    mu_delta_CO = # see experiment -> observables data

    alpha = pm.StudentT('alpha', nu=4, mu=mu_alpha, sigma=0.05, observed=alpha_obs)
    delta_OH = pm.StudentT('delta_OH', nu=4, mu=mu_delta_OH, sigma=0.1, observed=delta_OH_obs)
    delta_CO = pm.StudentT('delta_CO', nu=4, mu=mu_delta_CO, sigma=0.1, observed=delta_CO_obs)
