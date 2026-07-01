Description of the Bayesian workflow to go from experimental data to model fits and plots. 

Preprocessing: 
1) CO oxidation data for different materials at different CO partial pressures (P_CO), electrolyte (either acid or base) concentration (C_H or C_KOH), potential - with certain number of replicates. 
2) Current density data is available for several pressures (n_P), electrolyte concs (n_C) and trials (trials). 
3) Truncate current density data to a certain number of points after the minima of transfer coefficient (or don't truncate). Interpolate current density onto a coarser common grid. 
4) Convert current density into TOFs. 
5) Calculate experimental kinetic observables (delta_CO, delta_H (acid) / delta_OH (base), alpha). 
6) Save all data into experiments_mean: 
   1) experiments_mean is a dict with keys [(C_KOH, P_CO)]. 
   2) experiments_mean contains: 
      1) truncated_E - used in model for potential (1D vector)
      2) truncated_rate - rate averaged out over trials (1D vector) - can be an observed variable
      3) truncated_rate_SD - SD of rate over trials (1D vector) - can be SD for observed variable
      4) truncated_log_rate - log rate averaged out over trials (1D vector) - can be an observed variable
      5) truncated_log_rate_SD - SD of log rate over trials (1D vector) - can be SD for observed variable
      6) truncated_rate_matrix - set of arrays - rate of each trial (1D vector * trials) - observed variable in pymc models 
      7) E - full potential vector used for plotting rate, log_rate and alpha 
      8) rate - full averaged rate vector used for plotting 
      9) rate_SD - full rate SD used for plotting 
      10) log_rate - full averaged log rate vector used for plotting 
      11) log_rate_SD - full log rate SD used for plotting 
      12) alpha - full averaged transfer coefficients used for plotting
      13) alpha_SD - full transfer coefficient SD used for plotting 
      14) E_OH or E_H - potential vector used for plotting delta_OH (or delta_H)
      15) delta_OH or delta_H - averaged electrolyte conc order used for plotting 
      16) delta_OH_SD or delta_H_SD - electroluyte conc order SD used for plotting 
      17) E_CO - potential vector used for plotting delta_CO 
      18) delta_CO - averaged CO order used for plotting 
      19) delta_CO_SD - CO order SD used for plotting 
   3) SD vectors contain zeros if there is only one trial - this lets us use the same code even if we have one trial. 
   4) n_C and n_P >= 2 (so we don't have to take n_C = 1 or n_P = 1 cases into account). 
7) Notes about kinetic observables: 
   1) alpha is present for every C_KOH and P_CO 
   2) delta_OH / delta_H is only unique for different P_CO (duplicated for each C_OH/C_H in the experiments_mean dict). 
   3) delta_CO is present for C_KOH and (n_P - 1). delta_CO is missing for the highest partial pressure of CO (delta_CO is defined between consecutive CO pressures). 
8) experiments_mean is exported as a pickle file to be loaded into the model. 

Fitting: 
1) Inputs: C_KOH_in (or C_H_in), P_CO_in and E_in (all the same length as truncated_E). 
2) Pymc model is created and run - this part is trivial. Fitted variable in rate (truncated_rate_matrix in particular). Error model is used for predicted model SD. 
3) Post-processing: 
   1) Calcualte model kinetic observables (see notes about kinetic observables above for shape of KOs - length is same as E_in) (add_post_sampling_observables function)
   2) Plot posteriors and pair plots (plot_posteriors function) 
   3) Calculate flattened r2 (calculate_flattened_r2 function). Not technically the correct for non-linear models, but its a good qualitative metric. 
   4) PLot LOO-PIT and energy plot. 
   5) Plot model fits: 
      1) Plot pointiwise LOo-CV (summed up for replicates). Plot size: n_C columns, n_P rows.
      2) Plot rates (experimental (+exp SD) vs model (+ 95% CI from ppc)). Plot size: n_C columns, n_P rows. 
      3) Plot log rates (experimental (+exp SD) vs model (+ 95% CI from model parameter uncertainty)). Plot size: n_C columns, n_P rows. 
      4) Plot residual of log rates (experimental (+exp SD) vs model (+ 95% CI from model parameter uncertainty)). Plot size: n_C columns, n_P rows. 
      5) Plot alpha (experimental (+exp SD) vs model (+ 95% CI from model parameter uncertainty))
         1) Default plot size: n_C columns, n_P rows. 
         2) Consolidated plot size: n_C columns, 1 row. Each subplot has n_P pair of lines (experimental and model). SDs are not included in consolidated plot. 
      6) Plot delta_OH/delta_H (experimental (+exp SD) vs model (+ 95% CI from model parameter uncertainty))
         1) Default plot size: n_P columns, 1 row. 
         2) Consolidated plot size: 1 column, 1 row. Plot has n_P pair of lines (experimental and model). SDs are not included in consolidated plot. 
      7) Plot delta_CO (experimental (+exp SD) vs model (+ 95% CI from model parameter uncertainty))
         1) Default plot size: n_C columns, n_P-1 rows. 
         2) Consolidated plot size: n_C columns, 1 row. Each subplot has n_P-1 pair of lines (experimental and model). SDs are not included in consolidated plot. 
   6) Plot modeled surface coverages (saved as deterministic variables) (plot_coverages function - includes 95% CI from model parameter uncertainty)
   7) Calculate and plot DRCs (plot_drc function - includes 95% CI from model parameter uncertainty) (optional, only plotted when there are multiple rate controlling steps). 

