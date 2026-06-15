### Dual Site with Monofunctional and Bifunctional Langmuir-Hinshelwood with Lateral Interactions for CO as a function of CO coverage, and activation barrier from scaling/BEP ###

using PyCall
using NLsolve
using DifferentialEquations
using Statistics
using Random
using XLSX
using Polynomials
using Optim
using BlackBoxOptim
using DataFrames
using GLM
using LaTeXStrings
using TickTock
using StaticArrays
using ModelingToolkit
using StatsBase
using Distributions
using Printf
using LinearAlgebra

# setprecision(200)
mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
stats = pyimport("scipy.stats")
PyCall.fixqtpath()
np = pyimport("numpy")
mp = pyimport("mpmath")
scipy = pyimport("scipy")

function getExperimentalData(material)

    cd(@__DIR__)
    cd("..\\expt_data")
    data_file = XLSX.readxlsx(material * "_summary.xlsx")

    expt_rhe_potential = vec(convert(Array{Float64}, data_file["Sheet1"]["B2:B21"]))
    expt_she_potential = vec(convert(Array{Float64}, data_file["Sheet1"]["C2:C21"]))
    expt_transfer_coefficient_0pt1_250 = vec(convert(Array{Float64}, data_file["Sheet1"]["D2:D21"]))
    expt_transfer_coefficient_0pt1_250_err = vec(convert(Array{Float64}, data_file["Sheet1"]["E2:E21"]))
    expt_transfer_coefficient_0pt1_500 = vec(convert(Array{Float64}, data_file["Sheet1"]["F2:F21"]))
    expt_transfer_coefficient_0pt1_500_err = vec(convert(Array{Float64}, data_file["Sheet1"]["G2:G21"]))
    expt_transfer_coefficient_0pt1_1000 = vec(convert(Array{Float64}, data_file["Sheet1"]["H2:H21"]))
    expt_transfer_coefficient_0pt1_1000_err = vec(convert(Array{Float64}, data_file["Sheet1"]["I2:I21"]))
    expt_transfer_coefficient_1_250 = vec(convert(Array{Float64}, data_file["Sheet1"]["J2:J21"]))
    expt_transfer_coefficient_1_250_err = vec(convert(Array{Float64}, data_file["Sheet1"]["K2:K21"]))
    expt_transfer_coefficent_1_500 = vec(convert(Array{Float64}, data_file["Sheet1"]["L2:L21"]))
    expt_transfer_coefficent_1_500_err = vec(convert(Array{Float64}, data_file["Sheet1"]["M2:M21"]))
    expt_transfer_coefficient_1_1000 = vec(convert(Array{Float64}, data_file["Sheet1"]["N2:N21"]))
    expt_transfer_coefficient_1_1000_err = vec(convert(Array{Float64}, data_file["Sheet1"]["O2:O21"]))
    expt_transfer_coefficient_10_250 = vec(convert(Array{Float64}, data_file["Sheet1"]["P2:P21"]))
    expt_transfer_coefficient_10_250_err = vec(convert(Array{Float64}, data_file["Sheet1"]["Q2:Q21"]))
    expt_transfer_coefficient_10_500 = vec(convert(Array{Float64}, data_file["Sheet1"]["R2:R21"]))
    expt_transfer_coefficient_10_500_err = vec(convert(Array{Float64}, data_file["Sheet1"]["S2:S21"]))
    expt_transfer_coefficient_10_1000 = vec(convert(Array{Float64}, data_file["Sheet1"]["T2:T21"]))
    expt_transfer_coefficient_10_1000_err = vec(convert(Array{Float64}, data_file["Sheet1"]["U2:U21"]))
    expt_transfer_cofficient_100_250 = vec(convert(Array{Float64}, data_file["Sheet1"]["V2:V21"]))
    expt_transfer_cofficient_100_250_err = vec(convert(Array{Float64}, data_file["Sheet1"]["W2:W21"]))
    expt_transfer_coefficient_100_500 = vec(convert(Array{Float64}, data_file["Sheet1"]["X2:X21"]))
    expt_transfer_coefficient_100_500_err = vec(convert(Array{Float64}, data_file["Sheet1"]["Y2:Y21"]))
    expt_transfer_coefficient_100_1000 = vec(convert(Array{Float64}, data_file["Sheet1"]["Z2:Z21"]))
    expt_transfer_coefficient_100_1000_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AA2:AA21"]))
    expt_CO_order_250_0pt1_1 = vec(convert(Array{Float64}, data_file["Sheet1"]["AB2:AB21"]))
    expt_CO_order_250_0pt1_1_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AC2:AC21"]))
    expt_CO_order_250_1_10 = vec(convert(Array{Float64}, data_file["Sheet1"]["AD2:AD21"]))
    expt_CO_order_250_1_10_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AE2:AE21"]))
    expt_CO_order_250_10_100 = vec(convert(Array{Float64}, data_file["Sheet1"]["AF2:AF21"]))
    expt_CO_order_250_10_100_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AG2:AG21"]))
    expt_CO_order_500_0pt1_1 = vec(convert(Array{Float64}, data_file["Sheet1"]["AH2:AH21"]))
    expt_CO_order_500_0pt1_1_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AI2:AI21"]))
    expt_CO_order_500_1_10 = vec(convert(Array{Float64}, data_file["Sheet1"]["AJ2:AJ21"]))
    expt_CO_order_500_1_10_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AK2:AK21"]))
    expt_CO_order_500_10_100 = vec(convert(Array{Float64}, data_file["Sheet1"]["AL2:AL21"]))
    expt_CO_order_500_10_100_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AM2:AM21"]))
    expt_CO_order_1000_0pt1_1 = vec(convert(Array{Float64}, data_file["Sheet1"]["AN2:AN21"]))
    expt_CO_order_1000_0pt1_1_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AO2:AO21"]))
    expt_CO_order_1000_1_10 = vec(convert(Array{Float64}, data_file["Sheet1"]["AP2:AP21"]))
    expt_CO_order_1000_1_10_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AQ2:AQ21"]))
    expt_CO_order_1000_10_100 = vec(convert(Array{Float64}, data_file["Sheet1"]["AR2:AR21"]))
    expt_CO_order_1000_10_100_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AS2:AS21"]))
    expt_OH_order_0pt1 = vec(convert(Array{Float64}, data_file["Sheet1"]["AT2:AT21"]))
    expt_OH_order_0pt1_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AU2:AU21"]))
    expt_OH_order_1 = vec(convert(Array{Float64}, data_file["Sheet1"]["AV2:AV21"]))
    expt_OH_order_1_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AW2:AW21"]))
    expt_OH_order_10 = vec(convert(Array{Float64}, data_file["Sheet1"]["AX2:AX21"]))
    expt_OH_order_10_err = vec(convert(Array{Float64}, data_file["Sheet1"]["AY2:AY21"]))
    expt_OH_order_100 = vec(convert(Array{Float64}, data_file["Sheet1"]["AZ2:AZ21"]))
    expt_OH_order_100_err = vec(convert(Array{Float64}, data_file["Sheet1"]["BA2:BA21"]))

    return expt_rhe_potential, expt_she_potential, expt_transfer_coefficient_0pt1_250, expt_transfer_coefficient_0pt1_250_err, expt_transfer_coefficient_0pt1_500, expt_transfer_coefficient_0pt1_500_err, 
    expt_transfer_coefficient_0pt1_1000, expt_transfer_coefficient_0pt1_1000_err, expt_transfer_coefficient_1_250, expt_transfer_coefficient_1_250_err, expt_transfer_coefficent_1_500, 
    expt_transfer_coefficent_1_500_err, expt_transfer_coefficient_1_1000, expt_transfer_coefficient_1_1000_err, expt_transfer_coefficient_10_250, expt_transfer_coefficient_10_250_err,
    expt_transfer_coefficient_10_500, expt_transfer_coefficient_10_500_err, expt_transfer_coefficient_10_1000, expt_transfer_coefficient_10_1000_err, expt_transfer_cofficient_100_250,
    expt_transfer_cofficient_100_250_err, expt_transfer_coefficient_100_500, expt_transfer_coefficient_100_500_err, expt_transfer_coefficient_100_1000, expt_transfer_coefficient_100_1000_err,
    expt_CO_order_250_0pt1_1, expt_CO_order_250_0pt1_1_err, expt_CO_order_250_1_10, expt_CO_order_250_1_10_err, expt_CO_order_250_10_100, expt_CO_order_250_10_100_err, expt_CO_order_500_0pt1_1, 
    expt_CO_order_500_0pt1_1_err, expt_CO_order_500_1_10, expt_CO_order_500_1_10_err, expt_CO_order_500_10_100, expt_CO_order_500_10_100_err, expt_CO_order_1000_0pt1_1, expt_CO_order_1000_0pt1_1_err, 
    expt_CO_order_1000_1_10, expt_CO_order_1000_1_10_err, expt_CO_order_1000_10_100, expt_CO_order_1000_10_100_err, expt_OH_order_0pt1, expt_OH_order_0pt1_err, expt_OH_order_1, 
    expt_OH_order_1_err, expt_OH_order_10, expt_OH_order_10_err, expt_OH_order_100, expt_OH_order_100_err
end

material = "Ag90Pd10"
getExperimentalData_output = getExperimentalData(material)

### Write in the outputs of the new function
expt_rhe_potential = getExperimentalData_output[1]
expt_she_potential = getExperimentalData_output[2]
expt_transfer_coefficient_250_0pt1 = getExperimentalData_output[3]
expt_transfer_coefficient_250_0pt1_err = getExperimentalData_output[4]
expt_transfer_coefficient_500_0pt1 = getExperimentalData_output[5]
expt_transfer_coefficient_500_0pt1_err = getExperimentalData_output[6]
expt_transfer_coefficient_1000_0pt1 = getExperimentalData_output[7]
expt_transfer_coefficient_1000_0pt1_err = getExperimentalData_output[8]
expt_transfer_coefficient_250_1 = getExperimentalData_output[9]
expt_transfer_coefficient_250_1_err = getExperimentalData_output[10]
expt_transfer_coefficient_500_1 = getExperimentalData_output[11]
expt_transfer_coefficient_500_1_err = getExperimentalData_output[12]
expt_transfer_coefficient_1000_1 = getExperimentalData_output[13]
expt_transfer_coefficient_1000_1_err = getExperimentalData_output[14]
expt_transfer_coefficient_250_10 = getExperimentalData_output[15]
expt_transfer_coefficient_250_10_err = getExperimentalData_output[16]
expt_transfer_coefficient_500_10 = getExperimentalData_output[17]
expt_transfer_coefficient_500_10_err = getExperimentalData_output[18]
expt_transfer_coefficient_1000_10 = getExperimentalData_output[19]
expt_transfer_coefficient_1000_10_err = getExperimentalData_output[20]
expt_transfer_coefficient_250_100 = getExperimentalData_output[21]
expt_transfer_coefficient_250_100_err = getExperimentalData_output[22]
expt_transfer_coefficient_500_100 = getExperimentalData_output[23]
expt_transfer_coefficient_500_100_err = getExperimentalData_output[24]
expt_transfer_coefficient_1000_100 = getExperimentalData_output[25]
expt_transfer_coefficient_1000_100_err = getExperimentalData_output[26]
expt_CO_order_250_0pt1_1 = getExperimentalData_output[27]
expt_CO_order_250_0pt1_1_err = getExperimentalData_output[28]
expt_CO_order_250_1_10 = getExperimentalData_output[29]
expt_CO_order_250_1_10_err = getExperimentalData_output[30]
expt_CO_order_250_10_100 = getExperimentalData_output[31]
expt_CO_order_250_10_100_err = getExperimentalData_output[32]
expt_CO_order_500_0pt1_1 = getExperimentalData_output[33]
expt_CO_order_500_0pt1_1_err = getExperimentalData_output[34]
expt_CO_order_500_1_10 = getExperimentalData_output[35]
expt_CO_order_500_1_10_err = getExperimentalData_output[36]
expt_CO_order_500_10_100 = getExperimentalData_output[37]
expt_CO_order_500_10_100_err = getExperimentalData_output[38]
expt_CO_order_1000_0pt1_1 = getExperimentalData_output[39]
expt_CO_order_1000_0pt1_1_err = getExperimentalData_output[40]
expt_CO_order_1000_1_10 = getExperimentalData_output[41]
expt_CO_order_1000_1_10_err = getExperimentalData_output[42]
expt_CO_order_1000_10_100 = getExperimentalData_output[43]
expt_CO_order_1000_10_100_err = getExperimentalData_output[44]
expt_OH_order_0pt1 = getExperimentalData_output[45]
expt_OH_order_0pt1_err = getExperimentalData_output[46]
expt_OH_order_1 = getExperimentalData_output[47]
expt_OH_order_1_err = getExperimentalData_output[48]
expt_OH_order_10 = getExperimentalData_output[49]
expt_OH_order_10_err = getExperimentalData_output[50]
expt_OH_order_100 = getExperimentalData_output[51]
expt_OH_order_100_err = getExperimentalData_output[52]

println("Data imported successfully")

# We will simulate at many more potential points than we have experimental data, because this helps with updating the initial guesses for the MKM solver at each potential (might not be needed for these simple models, can decide whether to use this or not)
if material == "Pd100"
    E_init_simulate_dense = 0.7
    E_final_simulate_dense = 1
    E_step_simulate_dense = 0.01
    E_array_simulated = round.(np.arange(E_init_simulate_dense, E_final_simulate_dense, E_step_simulate_dense); digits = 4)
    E_array_OH_simulated = round.(np.arange(E_init_simulate_dense-0.8, E_final_simulate_dense-0.8, E_step_simulate_dense); digits = 4)
elseif material == "Ag10Pd90"
    E_init_simulate_dense = 0.7
    E_final_simulate_dense = 1
    E_step_simulate_dense = 0.01
    E_array_simulated = round.(np.arange(E_init_simulate_dense, E_final_simulate_dense, E_step_simulate_dense); digits = 4)
    E_array_OH_simulated = round.(np.arange(E_init_simulate_dense-0.8, E_final_simulate_dense-0.8, E_step_simulate_dense); digits = 4)
else
    E_init_simulate_dense = 0.55
    E_final_simulate_dense = 1
    E_step_simulate_dense = 0.01
    E_array_simulated = round.(np.arange(E_init_simulate_dense, E_final_simulate_dense, E_step_simulate_dense); digits = 4)
    E_array_OH_simulated = round.(np.arange(E_init_simulate_dense-0.8, E_final_simulate_dense-0.8, E_step_simulate_dense); digits = 4)
end

# Define rate constants in terms of standard free energies, can separate into enthalpy/entropy later if we want
kB = 8.6173e-5 # eV/K
h = 4.1357e-15 # eV*s
T = 292.15 # K (19 C, lab temperature)
Eref = 0
a_H2O = 1 # activity of H2O(l)
P_CO2 = 0.001 # pressure of CO2(g), should be very low due to differential conversion and "far from equilibrium" operation
G0_overall_ref = -0.21 # eV

function controlFitting(global_opt_iter, local_opt_iter, priors, bounds)
    # Unpack the priors here

    dG0_CO_g_to_CO_ref_prior_mean = priors[1]
    dG0_OH_aq_to_OH_ref_prior_mean = priors[2]
    z_CO_CO_prior_mean = priors[3]
    z_CO_OH_prior_mean = priors[4]
    z_OH_CO_prior_mean = priors[5]
    z_OH_OH_prior_mean = priors[6]

    dG0_CO_g_to_CO_ref_bounds = bounds[1]
    dG0_OH_aq_to_OH_ref_bounds = bounds[2]
    z_CO_CO_bounds = bounds[3]
    z_CO_OH_bounds = bounds[4]
    z_OH_CO_bounds = bounds[5]
    z_OH_OH_bounds = bounds[6]

    function simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, E_array, P_CO_array, a_OH_array, orderType)

        alpha_250 = zeros(length(P_CO_array),length(E_array))
        alpha_500 = zeros(length(P_CO_array),length(E_array))
        alpha_1000 = zeros(length(P_CO_array),length(E_array))
        CO_order_250 = zeros(3,length(E_array))
        CO_order_500 = zeros(3,length(E_array))
        CO_order_1000 = zeros(3,length(E_array))
        OH_order = zeros(length(P_CO_array),length(E_array))

        # Deal with the thermodynamic species first
        G0_CO_ref = dG0_CO_g_to_CO_ref
        G0_OH_ref = dG0_OH_aq_to_OH_ref

        # Initial condition first
        initialConditionFirst = [0.75, 0.01]

        ### BEP COEFFICIENTS AND SITE RATIOS
        alpha_BEP = 0.44
        gamma_BEP = 0.6

        if orderType == "CO order"
            t_CO_CO_order_dict = Dict()
            t_OH_CO_order_dict = Dict()
            total_rate_dict = Dict()
            for i = 1:length(a_OH_array)
                t_CO_array_for_CO_order = zeros(length(P_CO_array),length(E_array))
                t_OH_array_for_CO_order = zeros(length(P_CO_array),length(E_array))
                k_LH_fwd = zeros(length(E_array))
                ddG_LH = zeros(length(P_CO_array),length(E_array))
                k_LH_fwd_cov_dep = zeros(length(P_CO_array),length(E_array))
                LH_rate = zeros(length(P_CO_array),length(E_array))
                total_rate = zeros(length(P_CO_array),length(E_array))
                ln_rate_for_CO_order = zeros(length(P_CO_array),length(E_array))
            
                # Need to loop through E_array explicitly since we need to solve a system of eqns. at every potential (can't feed arrays into the NL solver)
                # Get the CO order by looping through potential (inner loop) and CO pressures (outer loop)
                for j = 1:length(P_CO_array)
                    for k = 1:length(E_array)
                        # Update the initial conditions using the solved coverages at the previous potential
                        if k == 1
                            IC = initialConditionFirst
                        else
                            IC = [t_CO_array_for_CO_order[j,k-1], t_OH_array_for_CO_order[j,k-1]] 
                        end

                        G0_CO = G0_CO_ref 
                        G0_OH = G0_OH_ref - (E_array[k] - Eref)
                        #define barriers at reference potential and zero coverage
                        if alpha_BEP + gamma_BEP * (0.67 - 0.25 * G0_CO_ref - G0_OH_ref) < 0
                            dG_LH_act_ref = 0
                        else
                            dG_LH_act_ref = alpha_BEP + gamma_BEP * (0.67 - 0.25 * G0_CO_ref - G0_OH_ref)
                        end

                        #define potential dependent barriers at zero coverage
                        dG_LH_act = dG_LH_act_ref

                        # Get the equilibrium constants in the limit of zero coverage
                        K_CO_0 = exp(-G0_CO/(kB*T))
                        K_OH_0 = exp(-G0_OH/(kB*T))

                        # And the potential-dependent forward ER rate constant
                        k_LH_fwd[k] = (kB*T/h) * exp(-dG_LH_act/(kB*T))

                        # Need to solve system of non-linear algebraic equations describing the coverages
                        function f!(fvec,x)
                            t_CO = x[1]
                            t_OH = x[2]
                            # Group the "driving force" terms for each species (i.e. the numerators in the expressions for the coverages of each species)
                            driving_CO = (K_CO_0 * exp(-(z_CO_CO * t_CO + z_CO_OH * t_OH)/(kB*T)) * P_CO_array[j])
                            driving_OH = (K_OH_0 * exp(-(z_OH_CO * t_CO + z_OH_OH * t_OH)/(kB*T)) * a_OH_array[i])

                            # All denominators corresponding to each site type are the same.
                            denom = 1 + driving_CO + driving_OH

                            # Set coverage terms equal to zero and solve for roots
                            fvec[1] = (t_CO * denom) - driving_CO
                            fvec[2] = (t_OH * denom) - driving_OH

                        end
                        function j!(jvec, x)
                            t_CO = x[1]
                            t_OH = x[2]
                            # Group the "driving force" terms for each species (i.e. the numerators in the expressions for the coverages of each species)
                            driving_CO = (K_CO_0 * exp(-(z_CO_CO * t_CO + z_CO_OH * t_OH)/(kB*T)) * P_CO_array[j])
                            driving_OH = (K_OH_0 * exp(-(z_OH_CO * t_CO + z_OH_OH * t_OH)/(kB*T)) * a_OH_array[i])

                            # All denominators corresponding to each site type are the same.
                            denom = 1 + driving_CO + driving_OH

                            jvec[1,1] = denom + z_CO_CO/(kB*T) * driving_CO * (1 - t_CO) - z_OH_CO/(kB*T) * driving_OH * t_CO
                            jvec[1,2] = z_CO_OH/(kB*T) * driving_CO * (1 - t_CO) - z_OH_OH/(kB*T) * driving_OH * t_CO
                            jvec[2,1] = z_OH_CO/(kB*T) * driving_OH * (1 - t_OH) - z_CO_CO/(kB*T) * driving_CO * t_OH
                            jvec[2,2] = denom + z_OH_OH/(kB*T) * driving_OH * (1 - t_OH) - z_CO_OH/(kB*T) * driving_CO * t_OH
                    
                        end
                        res = nlsolve(f!, j!, big.(IC), ftol = 1e-8, method=:newton)
                        if converged(res) == true
                            solution = res.zero
                        else
                            low_res = nlsolve(f!, j!, big.(IC), ftol = 1e-8, method=:trust_region, factor = 2)
                            solution = low_res.zero
                        end
                        if solution[1] < 0
                            t_CO_array_for_CO_order[j,k] = 10 ^ (-20)
                        elseif solution[1] > 1
                            t_CO_array_for_CO_order[j,k] = 0.99999999999999999999
                        else
                            t_CO_array_for_CO_order[j,k] = solution[1]
                        end
                        if solution[2] < 0
                            t_OH_array_for_CO_order[j,k] = 10 ^ (-20)
                        elseif solution[2] > 1
                            t_OH_array_for_CO_order[j,k] = 0.99999999999999999999
                        else
                            t_OH_array_for_CO_order[j,k] = solution[2]
                        end

                    end

                    ddG_LH[j,:] = -1 .* (gamma_BEP .* (z_OH_CO .* t_CO_array_for_CO_order[j,:] + z_OH_OH .* t_OH_array_for_CO_order[j,:]) + 0.25 .* gamma_BEP .* (z_CO_CO .* t_CO_array_for_CO_order[j,:] + z_CO_OH .* t_OH_array_for_CO_order[j,:]))

                    k_LH_fwd_cov_dep[j,:] = k_LH_fwd[:] .* exp.(-ddG_LH[j,:] ./ (kB*T))

                    LH_rate[j,:] = k_LH_fwd_cov_dep[j,:] .* t_CO_array_for_CO_order[j,:] .* t_OH_array_for_CO_order[j,:]

                    total_rate[j,:] = LH_rate[j,:]
                
                    ln_rate_for_CO_order[j,:] = log.(abs.(total_rate[j,:]))

                end
                t_CO_CO_order_dict[i] = t_CO_array_for_CO_order
                t_OH_CO_order_dict[i] = t_OH_array_for_CO_order
                total_rate_dict[i] = total_rate
                
                # Now actually evaluate the CO orders and transfer coefficient from the rates that were computed above.
                if i == 1
                    for k = 1:length(E_array)
                        CO_order_250[1,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[1:2]), Y = ln_rate_for_CO_order[1:2,k])))[2]
                        CO_order_250[2,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[2:3]), Y = ln_rate_for_CO_order[2:3,k])))[2]
                        CO_order_250[3,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[3:4]), Y = ln_rate_for_CO_order[3:4,k])))[2]
                    end
                elseif i == 2
                    for k = 1:length(E_array)
                        CO_order_500[1,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[1:2]), Y = ln_rate_for_CO_order[1:2,k])))[2]
                        CO_order_500[2,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[2:3]), Y = ln_rate_for_CO_order[2:3,k])))[2]
                        CO_order_500[3,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[3:4]), Y = ln_rate_for_CO_order[3:4,k])))[2]
                    end
                else
                    for k = 1:length(E_array)
                        CO_order_1000[1,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[1:2]), Y = ln_rate_for_CO_order[1:2,k])))[2]
                        CO_order_1000[2,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[2:3]), Y = ln_rate_for_CO_order[2:3,k])))[2]
                        CO_order_1000[3,k] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(P_CO_array[3:4]), Y = ln_rate_for_CO_order[3:4,k])))[2]
                    end
                end
                # Evaluate the apparent transfer coefficient at each of the CO pressures
                if i == 1
                    for j = 1:length(P_CO_array)
                        for k = 1:length(E_array)
                            if k == 1
                                alpha_250[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k]) / (E_array[k+1] - E_array[k])
                            elseif k == length(E_array)
                                alpha_250[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k] - ln_rate_for_CO_order[j,k-1]) / (E_array[k] - E_array[k-1])
                            else
                                alpha_250[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k-1]) / (E_array[k+1] - E_array[k-1])
                            end
                        end
                    end
                elseif i == 2
                    for j = 1:length(P_CO_array)
                        for k = 1:length(E_array)
                            if k == 1
                                alpha_500[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k]) / (E_array[k+1] - E_array[k])
                            elseif k == length(E_array)
                                alpha_500[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k] - ln_rate_for_CO_order[j,k-1]) / (E_array[k] - E_array[k-1])
                            else
                                alpha_500[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k-1]) / (E_array[k+1] - E_array[k-1])
                            end
                        end
                    end
                elseif i == 3
                    for j = 1:length(P_CO_array)
                        for k = 1:length(E_array)
                            if k == 1
                                alpha_1000[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k]) / (E_array[k+1] - E_array[k])
                            elseif k == length(E_array)
                                alpha_1000[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k] - ln_rate_for_CO_order[j,k-1]) / (E_array[k] - E_array[k-1])
                            else
                                alpha_1000[j,k] = (kB*T) * (ln_rate_for_CO_order[j,k+1] - ln_rate_for_CO_order[j,k-1]) / (E_array[k+1] - E_array[k-1])
                            end
                        end
                    end
                end
            end    
        end

        # Get the OH order by looping through potential (inner loop) and OH activity (outer loop)
        if orderType == "OH order"
            t_CO_OH_order_dict = Dict()
            t_OH_OH_order_dict = Dict()
            total_rate_dict = Dict()

            E_OH_ref = -0.8

            for i = 1:length(P_CO_array)
                # Need to loop through E_array explicitly since we need to solve a system of eqns. at every potential (can't feed arrays into the NL solver)
                t_CO_array_for_OH_order = zeros(length(a_OH_array),length(E_array))
                t_OH_array_for_OH_order = zeros(length(a_OH_array),length(E_array))
                k_LH_fwd = zeros(length(E_array))
                ddG_LH = zeros(length(a_OH_array),length(E_array))
                k_LH_fwd_cov_dep = zeros(length(a_OH_array),length(E_array))
                LH_rate = zeros(length(a_OH_array),length(E_array))
                total_rate = zeros(length(a_OH_array),length(E_array))
                ln_rate_for_OH_order = zeros(length(a_OH_array),length(E_array))
                # Get the OH order by looping through potential (inner loop) and OH activities (outer loop)
                for j = 1:length(a_OH_array)
                    for k = 1:length(E_array)
                        # Update the initial conditions using the solved coverages at the previous potential
                        if k == 1
                            IC = initialConditionFirst
                        else
                            IC = [t_CO_array_for_OH_order[j,k-1], t_OH_array_for_OH_order[j,k-1]] 
                        end

                        G0_CO = G0_CO_ref
                        G0_OH = G0_OH_ref - (E_array[k] - E_OH_ref)
                        #define barriers at reference potential and zero coverage
                        if alpha_BEP + gamma_BEP * (0.67 - 0.25 * G0_CO_ref - G0_OH_ref) < 0
                            dG_LH_act_ref = 0
                        else
                            dG_LH_act_ref = alpha_BEP + gamma_BEP * (0.67 - 0.25 * G0_CO_ref - G0_OH_ref)
                        end

                        #define potential dependent barriers at zero coverage
                        dG_LH_act = dG_LH_act_ref

                        # Get the equilibrium constants in the limit of zero coverage
                        K_CO_0 = exp(-G0_CO/(kB*T))
                        K_OH_0 = exp(-G0_OH/(kB*T))

                        # And the potential-dependent forward ER rate constant
                        k_LH_fwd[k] = (kB*T/h) * exp(-dG_LH_act/(kB*T))

                        # Need to solve system of non-linear algebraic equations describing the coverages
                        function f!(fvec,x)
                            t_CO = x[1]
                            t_OH = x[2]
                            # Group the "driving force" terms for each species (i.e. the numerators in the expressions for the coverages of each species)
                            driving_CO = (K_CO_0 * exp(-(z_CO_CO * t_CO + z_CO_OH * t_OH)/(kB*T)) * P_CO_array[i])
                            driving_OH = (K_OH_0 * exp(-(z_OH_CO * t_CO + z_OH_OH * t_OH)/(kB*T)) * a_OH_array[j])

                            # All denominators corresponding to each site type are the same.
                            denom = 1 + driving_CO + driving_OH

                            # Set coverage terms equal to zero and solve for roots
                            fvec[1] = (t_CO * denom) - driving_CO
                            fvec[2] = (t_OH * denom) - driving_OH
                        end
                        function j!(jvec, x)
                            t_CO = x[1]
                            t_OH = x[2]
                            # Group the "driving force" terms for each species (i.e. the numerators in the expressions for the coverages of each species)
                            driving_CO = (K_CO_0 * exp(-(z_CO_CO * t_CO + z_CO_OH * t_OH)/(kB*T)) * P_CO_array[i])
                            driving_OH = (K_OH_0 * exp(-(z_OH_CO * t_CO + z_OH_OH * t_OH)/(kB*T)) * a_OH_array[j])

                            # All denominators corresponding to each site type are the same.
                            denom = 1 + driving_CO + driving_OH

                            jvec[1,1] = denom + z_CO_CO/(kB*T) * driving_CO * (1 - t_CO) - z_OH_CO/(kB*T) * driving_OH * t_CO
                            jvec[1,2] = z_CO_OH/(kB*T) * driving_CO * (1 - t_CO) - z_OH_OH/(kB*T) * driving_OH * t_CO
                            jvec[2,1] = z_OH_CO/(kB*T) * driving_OH * (1 - t_OH) - z_CO_CO/(kB*T) * driving_CO * t_OH
                            jvec[2,2] = denom + z_OH_OH/(kB*T) * driving_OH * (1 - t_OH) - z_CO_OH/(kB*T) * driving_CO * t_OH
                        end
                        res = nlsolve(f!, j!, big.(IC), ftol = 1e-8, method=:newton)
                        if converged(res) == true
                            solution = res.zero
                        else
                            low_res = nlsolve(f!, j!, big.(IC), ftol = 1e-8, method=:trust_region, factor = 2)
                            solution = low_res.zero
                        end
                        if solution[1] < 0
                            t_CO_array_for_OH_order[j,k] = 10 ^ (-20)
                        elseif solution[1] > 1
                            t_CO_array_for_OH_order[j,k] = 0.99999999999999999999
                        else
                            t_CO_array_for_OH_order[j,k] = solution[1]
                        end
                        if solution[2] < 0
                            t_OH_array_for_OH_order[j,k] = 10 ^ (-20)
                        elseif solution[2] > 1
                            t_OH_array_for_OH_order[j,k] = 0.99999999999999999999
                        else
                            t_OH_array_for_OH_order[j,k] = solution[2]
                        end
                    end
                    ddG_LH[j,:] = -1 .* (gamma_BEP .* (z_OH_CO .* t_CO_array_for_OH_order[j,:] + z_OH_OH .* t_OH_array_for_OH_order[j,:]) + 0.25 .* gamma_BEP .* (z_CO_CO .* t_CO_array_for_OH_order[j,:] + z_CO_OH .* t_OH_array_for_OH_order[j,:]))

                    k_LH_fwd_cov_dep[j,:] = k_LH_fwd[:] .* exp.(-ddG_LH[j,:] ./ (kB*T))
                
                    LH_rate[j,:] = k_LH_fwd_cov_dep[j,:] .* t_CO_array_for_OH_order[j,:] .* t_OH_array_for_OH_order[j,:]

                    total_rate[j,:] = LH_rate[j,:]

                    ln_rate_for_OH_order[j,:] = log.(abs.(total_rate[j,:]))
                end
                # Now actually evaluate the OH orders and transfer coefficient from the rates that were computed above.
                for j = 1:length(E_array)
                    OH_order[i,j] = coef(lm(@formula(Y ~ X), DataFrame(X = log.(a_OH_array), Y = ln_rate_for_OH_order[:,j])))[2]
                end
                t_CO_OH_order_dict[i] = t_CO_array_for_OH_order
                t_OH_OH_order_dict[i] = t_OH_array_for_OH_order
                total_rate_dict[i] = total_rate
            end 
        end
        # Output the fully 2D arrays (coverages, alpha, etc) across the range of CO pressures
        if orderType == "CO order"
            return alpha_250, alpha_500, alpha_1000, CO_order_250, CO_order_500, CO_order_1000, t_CO_CO_order_dict, t_OH_CO_order_dict, total_rate_dict
        elseif orderType == "OH order"
            return OH_order, t_CO_OH_order_dict, t_OH_OH_order_dict, total_rate_dict
        end
    end

    function getNegLogLikelihood(params)

        dG0_CO_g_to_CO_ref = params[1]
        dG0_OH_aq_to_OH_ref = params[2]
        z_CO_CO = params[3]
        z_CO_OH = params[4]
        z_OH_CO = params[5]
        z_OH_OH = params[6]

        P_CO_array = [0.00092, 0.0092, 0.092, 0.92] # bar, 0.1%, 1%, 10%, 100% CO with total P typically around 0.92 bar (slight positive pressure in cell but lower atm pressure in Boulder)
        a_OH_array = [0.19, 0.37, 0.76] # activities corrected from concentrations

        model_output = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, expt_rhe_potential, P_CO_array, a_OH_array, "CO order")
        model_output_OH_order = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, expt_she_potential, P_CO_array, a_OH_array, "OH order")

        alpha_simulated_250 = model_output[1]
        # Extract transfer coefficients at the different CO pressures
        alpha_simulated_250_0_1_CO = alpha_simulated_250[1,:]
        alpha_simulated_250_1_CO = alpha_simulated_250[2,:]
        alpha_simulated_250_10_CO = alpha_simulated_250[3,:]
        alpha_simulated_250_100_CO = alpha_simulated_250[4,:]

        alpha_simulated_500 = model_output[2]
        # Extract transfer coefficients at the different CO pressures
        alpha_simulated_500_0_1_CO = alpha_simulated_500[1,:]
        alpha_simulated_500_1_CO = alpha_simulated_500[2,:]
        alpha_simulated_500_10_CO = alpha_simulated_500[3,:]
        alpha_simulated_500_100_CO = alpha_simulated_500[4,:]

        alpha_simulated_1000 = model_output[3]
        # Extract transfer coefficients at the different CO pressures
        alpha_simulated_1000_0_1_CO = alpha_simulated_1000[1,:]
        alpha_simulated_1000_1_CO = alpha_simulated_1000[2,:]
        alpha_simulated_1000_10_CO = alpha_simulated_1000[3,:]
        alpha_simulated_1000_100_CO = alpha_simulated_1000[4,:]

        CO_order_simulated_250 = model_output[4]
        CO_order_simulated_250_0pt1_1 = CO_order_simulated_250[1,:]
        CO_order_simulated_250_1_10 = CO_order_simulated_250[2,:]
        CO_order_simulated_250_10_100 = CO_order_simulated_250[3,:]

        CO_order_simulated_500 = model_output[5]
        CO_order_simulated_500_0pt1_1 = CO_order_simulated_500[1,:]
        CO_order_simulated_500_1_10 = CO_order_simulated_500[2,:]
        CO_order_simulated_500_10_100 = CO_order_simulated_500[3,:]

        CO_order_simulated_1000 = model_output[6]
        CO_order_simulated_1000_0pt1_1 = CO_order_simulated_1000[1,:]
        CO_order_simulated_1000_1_10 = CO_order_simulated_1000[2,:]
        CO_order_simulated_1000_10_100 = CO_order_simulated_1000[3,:]

        OH_order_simulated = model_output_OH_order[1]
        OH_order_simulated_0pt1 = OH_order_simulated[1,:]
        OH_order_simulated_1 = OH_order_simulated[2,:]
        OH_order_simulated_10 = OH_order_simulated[3,:]
        OH_order_simulated_100 = OH_order_simulated[4,:]

        # Run simultaneous fits across all of the CO pressure data points
        log_likelihood_alpha_250_0_1_CO = zeros(length(alpha_simulated_250_0_1_CO))
        log_likelihood_alpha_250_1_CO = zeros(length(alpha_simulated_250_1_CO))
        log_likelihood_alpha_250_10_CO = zeros(length(alpha_simulated_250_10_CO))
        log_likelihood_alpha_250_100_CO = zeros(length(alpha_simulated_250_100_CO))

        log_likelihood_alpha_500_0_1_CO = zeros(length(alpha_simulated_500_0_1_CO))
        log_likelihood_alpha_500_1_CO = zeros(length(alpha_simulated_500_1_CO))
        log_likelihood_alpha_500_10_CO = zeros(length(alpha_simulated_500_10_CO))
        log_likelihood_alpha_500_100_CO = zeros(length(alpha_simulated_500_100_CO))

        log_likelihood_alpha_1000_0_1_CO = zeros(length(alpha_simulated_1000_0_1_CO))
        log_likelihood_alpha_1000_1_CO = zeros(length(alpha_simulated_1000_1_CO))
        log_likelihood_alpha_1000_10_CO = zeros(length(alpha_simulated_1000_10_CO))
        log_likelihood_alpha_1000_100_CO = zeros(length(alpha_simulated_1000_100_CO))

        log_likelihood_CO_order_250_0pt1_1 = zeros(length(CO_order_simulated_250_0pt1_1))
        log_likelihood_CO_order_250_1_10 = zeros(length(CO_order_simulated_250_1_10))
        log_likelihood_CO_order_250_10_100 = zeros(length(CO_order_simulated_250_10_100))

        log_likelihood_CO_order_500_0pt1_1 = zeros(length(CO_order_simulated_500_0pt1_1))
        log_likelihood_CO_order_500_1_10 = zeros(length(CO_order_simulated_500_1_10))
        log_likelihood_CO_order_500_10_100 = zeros(length(CO_order_simulated_500_10_100))

        log_likelihood_CO_order_1000_0pt1_1 = zeros(length(CO_order_simulated_1000_0pt1_1))
        log_likelihood_CO_order_1000_1_10 = zeros(length(CO_order_simulated_1000_1_10))
        log_likelihood_CO_order_1000_10_100 = zeros(length(CO_order_simulated_1000_10_100))

        log_likelihood_OH_order_0pt1 = zeros(length(OH_order_simulated_0pt1))
        log_likelihood_OH_order_1 = zeros(length(OH_order_simulated_1))
        log_likelihood_OH_order_10 = zeros(length(OH_order_simulated_10))
        log_likelihood_OH_order_100 = zeros(length(OH_order_simulated_100))

        # Get the total log likelihood
        for i = 1:length(log_likelihood_OH_order_100)

            residuals_alpha_250_0_1_CO = (expt_transfer_coefficient_250_0pt1[i] - alpha_simulated_250_0_1_CO[i])^2
            log_likelihood_alpha_250_0_1_CO[i] = (-residuals_alpha_250_0_1_CO/(2*(expt_transfer_coefficient_250_0pt1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_250_0pt1_err[i]^2)))

            residuals_alpha_250_1_CO = (expt_transfer_coefficient_250_1[i] - alpha_simulated_250_1_CO[i])^2
            log_likelihood_alpha_250_1_CO[i] = (-residuals_alpha_250_1_CO/(2*(expt_transfer_coefficient_250_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_250_1_err[i]^2)))

            residuals_alpha_250_10_CO = (expt_transfer_coefficient_250_10[i] - alpha_simulated_250_10_CO[i])^2
            log_likelihood_alpha_250_10_CO[i] = (-residuals_alpha_250_10_CO/(2*(expt_transfer_coefficient_250_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_250_10_err[i]^2)))

            residuals_alpha_250_100_CO = (expt_transfer_coefficient_250_100[i] - alpha_simulated_250_100_CO[i])^2
            log_likelihood_alpha_250_100_CO[i] = (-residuals_alpha_250_100_CO/(2*(expt_transfer_coefficient_250_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_250_100_err[i]^2)))

            residuals_alpha_500_0_1_CO = (expt_transfer_coefficient_500_0pt1[i] - alpha_simulated_500_0_1_CO[i])^2
            log_likelihood_alpha_500_0_1_CO[i] = (-residuals_alpha_500_0_1_CO/(2*(expt_transfer_coefficient_500_0pt1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_500_0pt1_err[i]^2)))

            residuals_alpha_500_1_CO = (expt_transfer_coefficient_500_1[i] - alpha_simulated_500_1_CO[i])^2
            log_likelihood_alpha_500_1_CO[i] = (-residuals_alpha_500_1_CO/(2*(expt_transfer_coefficient_500_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_500_1_err[i]^2)))

            residuals_alpha_500_10_CO = (expt_transfer_coefficient_500_10[i] - alpha_simulated_500_10_CO[i])^2
            log_likelihood_alpha_500_10_CO[i] = (-residuals_alpha_500_10_CO/(2*(expt_transfer_coefficient_500_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_500_10_err[i]^2)))

            residuals_alpha_500_100_CO = (expt_transfer_coefficient_500_100[i] - alpha_simulated_500_100_CO[i])^2
            log_likelihood_alpha_500_100_CO[i] = (-residuals_alpha_500_100_CO/(2*(expt_transfer_coefficient_500_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_500_100_err[i]^2)))

            residuals_alpha_1000_0_1_CO = (expt_transfer_coefficient_1000_0pt1[i] - alpha_simulated_1000_0_1_CO[i])^2
            log_likelihood_alpha_1000_0_1_CO[i] = (-residuals_alpha_1000_0_1_CO/(2*(expt_transfer_coefficient_1000_0pt1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_1000_0pt1_err[i]^2)))

            residuals_alpha_1000_1_CO = (expt_transfer_coefficient_1000_1[i] - alpha_simulated_1000_1_CO[i])^2
            log_likelihood_alpha_1000_1_CO[i] = (-residuals_alpha_1000_1_CO/(2*(expt_transfer_coefficient_1000_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_1000_1_err[i]^2)))

            residuals_alpha_1000_10_CO = (expt_transfer_coefficient_1000_10[i] - alpha_simulated_1000_10_CO[i])^2
            log_likelihood_alpha_500_10_CO[i] = (-residuals_alpha_500_10_CO/(2*(expt_transfer_coefficient_500_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_500_10_err[i]^2)))

            residuals_alpha_1000_100_CO = (expt_transfer_coefficient_1000_100[i] - alpha_simulated_1000_100_CO[i])^2
            log_likelihood_alpha_1000_100_CO[i] = (-residuals_alpha_1000_100_CO/(2*(expt_transfer_coefficient_1000_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_transfer_coefficient_1000_100_err[i]^2)))

            residuals_CO_order_250_0pt1_1 = (expt_CO_order_250_0pt1_1[i] - CO_order_simulated_250_0pt1_1[i])^2
            log_likelihood_CO_order_250_0pt1_1[i] = (-residuals_CO_order_250_0pt1_1/(2*(expt_CO_order_250_0pt1_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_250_0pt1_1_err[i]^2)))

            residuals_CO_order_250_1_10 = (expt_CO_order_250_1_10[i] - CO_order_simulated_250_1_10[i])^2
            log_likelihood_CO_order_250_1_10[i] = (-residuals_CO_order_250_1_10/(2*(expt_CO_order_250_1_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_250_1_10_err[i]^2)))

            residuals_CO_order_250_10_100 = (expt_CO_order_250_10_100[i] - CO_order_simulated_250_10_100[i])^2
            log_likelihood_CO_order_250_10_100[i] = (-residuals_CO_order_250_10_100/(2*(expt_CO_order_250_10_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_250_10_100_err[i]^2)))

            residuals_CO_order_500_0pt1_1 = (expt_CO_order_500_0pt1_1[i] - CO_order_simulated_500_0pt1_1[i])^2
            log_likelihood_CO_order_500_0pt1_1[i] = (-residuals_CO_order_500_0pt1_1/(2*(expt_CO_order_500_0pt1_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_500_0pt1_1_err[i]^2)))

            residuals_CO_order_500_1_10 = (expt_CO_order_500_1_10[i] - CO_order_simulated_500_1_10[i])^2
            log_likelihood_CO_order_500_1_10[i] = (-residuals_CO_order_500_1_10/(2*(expt_CO_order_500_1_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_500_1_10_err[i]^2)))

            residuals_CO_order_500_10_100 = (expt_CO_order_500_10_100[i] - CO_order_simulated_500_10_100[i])^2
            log_likelihood_CO_order_500_10_100[i] = (-residuals_CO_order_500_10_100/(2*(expt_CO_order_500_10_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_500_10_100_err[i]^2)))

            residuals_CO_order_1000_0pt1_1 = (expt_CO_order_1000_0pt1_1[i] - CO_order_simulated_1000_0pt1_1[i])^2
            log_likelihood_CO_order_1000_0pt1_1[i] = (-residuals_CO_order_1000_0pt1_1/(2*(expt_CO_order_1000_0pt1_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_1000_0pt1_1_err[i]^2)))

            residuals_CO_order_1000_1_10 = (expt_CO_order_1000_1_10[i] - CO_order_simulated_1000_1_10[i])^2
            log_likelihood_CO_order_1000_1_10[i] = (-residuals_CO_order_1000_1_10/(2*(expt_CO_order_1000_1_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_1000_1_10_err[i]^2)))

            residuals_CO_order_1000_10_100 = (expt_CO_order_1000_10_100[i] - CO_order_simulated_1000_10_100[i])^2
            log_likelihood_CO_order_1000_10_100[i] = (-residuals_CO_order_1000_10_100/(2*(expt_CO_order_1000_10_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_CO_order_1000_10_100_err[i]^2)))

            residuals_OH_order_0pt1 = (expt_OH_order_0pt1[i] - OH_order_simulated_0pt1[i])^2
            log_likelihood_OH_order_0pt1[i] = (-residuals_OH_order_0pt1/(2*(expt_OH_order_0pt1_err[i]^2))) - (0.5 * log(2 * pi * (expt_OH_order_0pt1_err[i]^2)))

            residuals_OH_order_1 = (expt_OH_order_1[i] - OH_order_simulated_1[i])^2
            log_likelihood_OH_order_1[i] = (-residuals_OH_order_1/(2*(expt_OH_order_1_err[i]^2))) - (0.5 * log(2 * pi * (expt_OH_order_1_err[i]^2)))

            residuals_OH_order_10 = (expt_OH_order_10[i] - OH_order_simulated_10[i])^2
            log_likelihood_OH_order_10[i] = (-residuals_OH_order_10/(2*(expt_OH_order_10_err[i]^2))) - (0.5 * log(2 * pi * (expt_OH_order_10_err[i]^2)))

            residuals_OH_order_100 = (expt_OH_order_100[i] - OH_order_simulated_100[i])^2
            log_likelihood_OH_order_100[i] = (-residuals_OH_order_100/(2*(expt_OH_order_100_err[i]^2))) - (0.5 * log(2 * pi * (expt_OH_order_100_err[i]^2)))
        
        end

        total_log_likelihood_alpha_250_0_1_CO = sum(log_likelihood_alpha_250_0_1_CO)
        total_log_likelihood_alpha_250_1_CO = sum(log_likelihood_alpha_250_1_CO)
        total_log_likelihood_alpha_250_10_CO = sum(log_likelihood_alpha_250_10_CO)
        total_log_likelihood_alpha_250_100_CO = sum(log_likelihood_alpha_250_100_CO)

        total_log_likelihood_alpha_500_0_1_CO = sum(log_likelihood_alpha_500_0_1_CO)
        total_log_likelihood_alpha_500_1_CO = sum(log_likelihood_alpha_500_1_CO)
        total_log_likelihood_alpha_500_10_CO = sum(log_likelihood_alpha_500_10_CO)
        total_log_likelihood_alpha_500_100_CO = sum(log_likelihood_alpha_500_100_CO)

        total_log_likelihood_alpha_1000_0_1_CO = sum(log_likelihood_alpha_1000_0_1_CO)
        total_log_likelihood_alpha_1000_1_CO = sum(log_likelihood_alpha_1000_1_CO)
        total_log_likelihood_alpha_1000_10_CO = sum(log_likelihood_alpha_1000_10_CO)
        total_log_likelihood_alpha_1000_100_CO = sum(log_likelihood_alpha_1000_100_CO)

        total_log_likelihood_CO_order_250_0pt1_1 = sum(log_likelihood_CO_order_250_0pt1_1)
        total_log_likelihood_CO_order_250_1_10 = sum(log_likelihood_CO_order_250_1_10)
        total_log_likelihood_CO_order_250_10_100 = sum(log_likelihood_CO_order_250_10_100)

        total_log_likelihood_CO_order_500_0pt1_1 = sum(log_likelihood_CO_order_500_0pt1_1)
        total_log_likelihood_CO_order_500_1_10 = sum(log_likelihood_CO_order_500_1_10)
        total_log_likelihood_CO_order_500_10_100 = sum(log_likelihood_CO_order_500_10_100)

        total_log_likelihood_CO_order_1000_0pt1_1 = sum(log_likelihood_CO_order_1000_0pt1_1)
        total_log_likelihood_CO_order_1000_1_10 = sum(log_likelihood_CO_order_1000_1_10)
        total_log_likelihood_CO_order_1000_10_100 = sum(log_likelihood_CO_order_1000_10_100)

        total_log_likelihood_OH_order_0pt1 = sum(log_likelihood_OH_order_0pt1)
        total_log_likelihood_OH_order_1 = sum(log_likelihood_OH_order_1)
        total_log_likelihood_OH_order_10 = sum(log_likelihood_OH_order_10)
        total_log_likelihood_OH_order_100 = sum(log_likelihood_OH_order_100)

        total_neg_log_likelihood = sum([total_log_likelihood_alpha_250_0_1_CO, total_log_likelihood_alpha_250_1_CO, total_log_likelihood_alpha_250_10_CO, total_log_likelihood_alpha_250_100_CO, 
                                        total_log_likelihood_alpha_500_0_1_CO, total_log_likelihood_alpha_500_1_CO, total_log_likelihood_alpha_500_10_CO, total_log_likelihood_alpha_500_100_CO, 
                                        total_log_likelihood_alpha_1000_0_1_CO, total_log_likelihood_alpha_1000_1_CO, total_log_likelihood_alpha_1000_10_CO, total_log_likelihood_alpha_1000_100_CO, 
                                        total_log_likelihood_CO_order_250_0pt1_1, total_log_likelihood_CO_order_250_1_10, total_log_likelihood_CO_order_250_10_100, 
                                        total_log_likelihood_CO_order_500_0pt1_1, total_log_likelihood_CO_order_500_1_10, total_log_likelihood_CO_order_500_10_100, 
                                        total_log_likelihood_CO_order_1000_0pt1_1, total_log_likelihood_CO_order_1000_1_10, total_log_likelihood_CO_order_1000_10_100, 
                                        total_log_likelihood_OH_order_0pt1, total_log_likelihood_OH_order_1, total_log_likelihood_OH_order_10, total_log_likelihood_OH_order_100]) * -1
        #total_neg_log_likelihood = sum([total_log_likelihood_alpha_10_CO, total_log_likelihood_CO_order, total_log_likelihood_OH_order]) * -1
        #println(total_neg_log_likelihood)
        if isnan(total_neg_log_likelihood) == true
            total_neg_log_likelihood = 10^10
        else
        end
        return total_neg_log_likelihood
    end

    function runOptimization(objective)

        # initialGuess
        initialGuess = [dG0_CO_g_to_CO_ref_prior_mean, 
                        dG0_OH_aq_to_OH_ref_prior_mean, 
                        z_CO_CO_prior_mean, 
                        z_CO_OH_prior_mean, 
                        z_OH_CO_prior_mean, 
                        z_OH_OH_prior_mean]

        # bnds
        bnds = (dG0_CO_g_to_CO_ref_bounds, dG0_OH_aq_to_OH_ref_bounds, z_CO_CO_bounds, z_CO_OH_bounds, z_OH_CO_bounds, z_OH_OH_bounds)

        minimizer_opt = Dict("maxiter" => local_opt_iter)
        minimizer_method = Dict("method" => "L-BFGS-B", "bounds" => bnds, "options" => minimizer_opt)

        local_mins = zeros(global_opt_iter + 1, length(initialGuess) + 1)
        count = 1
        
        function callback(x, f, accepted)
            for i = 1:length(x)
                local_mins[count, i] = x[i]
                @printf("Parameter %s = %.3f ", i, x[i])
                println()
            end
            local_mins[count, length(x) + 1] = f
            count = count + 1
            return
        end
        println("Beginning Optimatization Routine")
        res = scipy.optimize.basinhopping(objective, minimizer_kwargs = minimizer_method, initialGuess, niter = global_opt_iter, disp = "True", niter_success = 100, callback = callback)
        #res = scipy.optimize.basinhopping(objective, minimizer_kwargs = minimizer_method, initialGuess, niter = 2, disp = "True", callback = callback)

        return res, local_mins

    end

    function simulateWithFittedParams(fitted_params)

        P_CO_array = [0.00092, 0.0092, 0.092, 0.92] # bar, 0.1%, 1%, 10%, 100% CO with total P typically around 0.92 bar (slight positive pressure in cell but lower atm pressure in Boulder)
        a_OH_array = [0.19, 0.37, 0.76] # activities corrected from concentrations

        dG0_CO_g_to_CO_ref = fitted_params[1]
        dG0_OH_aq_to_OH_ref = fitted_params[2]
        z_CO_CO = fitted_params[3]
        z_CO_OH = fitted_params[4]
        z_OH_CO = fitted_params[5]
        z_OH_OH = fitted_params[6]

        alpha_and_CO_order_simulatedOutput = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, E_array_simulated, P_CO_array, a_OH_array, "CO order")

        OH_order_simulatedOutput = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, E_array_OH_simulated, P_CO_array, a_OH_array, "OH order")

        # This is a 2D array of transfer coefficients across all of the CO pressures
        alpha_250_fitted = alpha_and_CO_order_simulatedOutput[1]
        alpha_500_fitted = alpha_and_CO_order_simulatedOutput[2]
        alpha_1000_fitted = alpha_and_CO_order_simulatedOutput[3]

        CO_order_250_fitted = alpha_and_CO_order_simulatedOutput[4]
        CO_order_500_fitted = alpha_and_CO_order_simulatedOutput[5]
        CO_order_1000_fitted = alpha_and_CO_order_simulatedOutput[6]

        OH_order_fitted = OH_order_simulatedOutput[1]

        # These coverage arrays will also be 2D evaluated over several CO pressures
        t_CO_CO_order = alpha_and_CO_order_simulatedOutput[7]
        t_CO_OH_order = OH_order_simulatedOutput[2]

        t_OH_CO_order = alpha_and_CO_order_simulatedOutput[8]
        t_OH_OH_order = OH_order_simulatedOutput[3]

        total_rate_CO_order = alpha_and_CO_order_simulatedOutput[9]
        total_rate_OH_order = OH_order_simulatedOutput[4]

        return t_CO_CO_order, t_CO_OH_order, t_OH_CO_order, t_OH_OH_order, alpha_250_fitted, alpha_500_fitted, alpha_1000_fitted, CO_order_250_fitted, CO_order_500_fitted, CO_order_1000_fitted, OH_order_fitted, total_rate_CO_order, total_rate_OH_order

    end

    function getChiSquared(fitted_params)

        P_CO_array = [0.00092, 0.0092, 0.092, 0.92] # bar, 0.1%, 1%, 10%, 100% CO with total P typically around 0.92 bar (slight positive pressure in cell but lower atm pressure in Boulder)
        a_OH_array = [0.19, 0.37, 0.76] # activities corrected from concentrations

        dG0_CO_g_to_CO_ref = fitted_params[1]
        dG0_OH_aq_to_OH_ref = fitted_params[2]
        z_CO_CO = fitted_params[3]
        z_CO_OH = fitted_params[4]
        z_OH_CO = fitted_params[5]
        z_OH_OH = fitted_params[6]

        alpha_and_CO_order_chi_sq = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, expt_rhe_potential, P_CO_array, a_OH_array, "CO order")

        OH_order_chi_sq = simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, z_CO_CO, z_CO_OH, z_OH_CO, z_OH_OH, expt_she_potential, P_CO_array, a_OH_array, "OH order")

        alpha_fitted_250_chi_sq = alpha_and_CO_order_chi_sq[1]
        alpha_fitted_250_0_1_CO_chi_sq = alpha_fitted_250_chi_sq[1,:]
        alpha_fitted_250_1_CO_chi_sq = alpha_fitted_250_chi_sq[2,:]
        alpha_fitted_250_10_CO_chi_sq = alpha_fitted_250_chi_sq[3,:]
        alpha_fitted_250_100_CO_chi_sq = alpha_fitted_250_chi_sq[4,:]

        alpha_fitted_500_chi_sq = alpha_and_CO_order_chi_sq[2]
        alpha_fitted_500_0_1_CO_chi_sq = alpha_fitted_500_chi_sq[1,:]
        alpha_fitted_500_1_CO_chi_sq = alpha_fitted_500_chi_sq[2,:]
        alpha_fitted_500_10_CO_chi_sq = alpha_fitted_500_chi_sq[3,:]
        alpha_fitted_500_100_CO_chi_sq = alpha_fitted_500_chi_sq[4,:]

        alpha_fitted_1000_chi_sq = alpha_and_CO_order_chi_sq[3]
        alpha_fitted_1000_0_1_CO_chi_sq = alpha_fitted_1000_chi_sq[1,:]
        alpha_fitted_1000_1_CO_chi_sq = alpha_fitted_1000_chi_sq[2,:]
        alpha_fitted_1000_10_CO_chi_sq = alpha_fitted_1000_chi_sq[3,:]
        alpha_fitted_1000_100_CO_chi_sq = alpha_fitted_1000_chi_sq[4,:]

        CO_order_fitted_250_chi_sq = alpha_and_CO_order_chi_sq[4]
        CO_order_fitted_250_0pt1_1_chi_sq = CO_order_fitted_250_chi_sq[1,:]
        CO_order_fitted_250_1_10_chi_sq = CO_order_fitted_250_chi_sq[2,:]
        CO_order_fitted_250_10_100_chi_sq = CO_order_fitted_250_chi_sq[3,:]

        CO_order_fitted_500_chi_sq = alpha_and_CO_order_chi_sq[5]
        CO_order_fitted_500_0pt1_1_chi_sq = CO_order_fitted_500_chi_sq[1,:]
        CO_order_fitted_500_1_10_chi_sq = CO_order_fitted_500_chi_sq[2,:]
        CO_order_fitted_500_10_100_chi_sq = CO_order_fitted_500_chi_sq[3,:]

        CO_order_fitted_1000_chi_sq = alpha_and_CO_order_chi_sq[6]
        CO_order_fitted_1000_0pt1_1_chi_sq = CO_order_fitted_1000_chi_sq[1,:]
        CO_order_fitted_1000_1_10_chi_sq = CO_order_fitted_1000_chi_sq[2,:]
        CO_order_fitted_1000_10_100_chi_sq = CO_order_fitted_1000_chi_sq[3,:]

        OH_order_fitted_chi_sq = OH_order_chi_sq[1]
        OH_order_fitted_0pt1_chi_sq = OH_order_fitted_chi_sq[1,:]
        OH_order_fitted_1_chi_sq = OH_order_fitted_chi_sq[2,:]
        OH_order_fitted_10_chi_sq = OH_order_fitted_chi_sq[3,:]
        OH_order_fitted_100_chi_sq = OH_order_fitted_chi_sq[4,:]
        
        chi_sq_alpha_250_0_1 = zeros(length(alpha_fitted_250_0_1_CO_chi_sq))
        chi_sq_alpha_250_1 = zeros(length(alpha_fitted_250_1_CO_chi_sq))
        chi_sq_alpha_250_10 = zeros(length(alpha_fitted_250_10_CO_chi_sq))
        chi_sq_alpha_250_100 = zeros(length(alpha_fitted_250_100_CO_chi_sq))

        chi_sq_alpha_500_0_1 = zeros(length(alpha_fitted_500_0_1_CO_chi_sq))
        chi_sq_alpha_500_1 = zeros(length(alpha_fitted_500_1_CO_chi_sq))
        chi_sq_alpha_500_10 = zeros(length(alpha_fitted_500_10_CO_chi_sq))
        chi_sq_alpha_500_100 = zeros(length(alpha_fitted_500_100_CO_chi_sq))

        chi_sq_alpha_1000_0_1 = zeros(length(alpha_fitted_1000_0_1_CO_chi_sq))
        chi_sq_alpha_1000_1 = zeros(length(alpha_fitted_1000_1_CO_chi_sq))
        chi_sq_alpha_1000_10 = zeros(length(alpha_fitted_1000_10_CO_chi_sq))
        chi_sq_alpha_1000_100 = zeros(length(alpha_fitted_1000_100_CO_chi_sq))

        chi_sq_CO_order_250_0pt1_1 = zeros(length(CO_order_fitted_250_0pt1_1_chi_sq))
        chi_sq_CO_order_250_1_10 = zeros(length(CO_order_fitted_250_1_10_chi_sq))
        chi_sq_CO_order_250_10_100 = zeros(length(CO_order_fitted_250_10_100_chi_sq))

        chi_sq_CO_order_500_0pt1_1 = zeros(length(CO_order_fitted_500_0pt1_1_chi_sq))
        chi_sq_CO_order_500_1_10 = zeros(length(CO_order_fitted_500_1_10_chi_sq))
        chi_sq_CO_order_500_10_100 = zeros(length(CO_order_fitted_500_10_100_chi_sq))

        chi_sq_CO_order_1000_0pt1_1 = zeros(length(CO_order_fitted_1000_0pt1_1_chi_sq))
        chi_sq_CO_order_1000_1_10 = zeros(length(CO_order_fitted_1000_1_10_chi_sq))
        chi_sq_CO_order_1000_10_100 = zeros(length(CO_order_fitted_1000_10_100_chi_sq))

        chi_sq_OH_order_0pt1 = zeros(length(OH_order_fitted_0pt1_chi_sq))
        chi_sq_OH_order_1 = zeros(length(OH_order_fitted_1_chi_sq))
        chi_sq_OH_order_10 = zeros(length(OH_order_fitted_10_chi_sq))
        chi_sq_OH_order_100 = zeros(length(OH_order_fitted_100_chi_sq))

        for i = 1:length(chi_sq_alpha_250_0_1)
            chi_sq_alpha_250_0_1[i] = ((expt_transfer_coefficient_250_0pt1[i] - alpha_fitted_250_0_1_CO_chi_sq[i])/expt_transfer_coefficient_250_0pt1_err[i])^2
            chi_sq_alpha_250_1[i] = ((expt_transfer_coefficient_250_1[i] - alpha_fitted_250_1_CO_chi_sq[i])/expt_transfer_coefficient_250_1_err[i])^2
            chi_sq_alpha_250_10[i] = ((expt_transfer_coefficient_250_10[i] - alpha_fitted_250_10_CO_chi_sq[i])/expt_transfer_coefficient_250_10_err[i])^2
            chi_sq_alpha_250_100[i] = ((expt_transfer_coefficient_250_100[i] - alpha_fitted_250_100_CO_chi_sq[i])/expt_transfer_coefficient_250_100_err[i])^2

            chi_sq_alpha_500_0_1[i] = ((expt_transfer_coefficient_500_0pt1[i] - alpha_fitted_500_0_1_CO_chi_sq[i])/expt_transfer_coefficient_500_0pt1_err[i])^2
            chi_sq_alpha_500_1[i] = ((expt_transfer_coefficient_500_1[i] - alpha_fitted_500_1_CO_chi_sq[i])/expt_transfer_coefficient_500_1_err[i])^2
            chi_sq_alpha_500_10[i] = ((expt_transfer_coefficient_500_10[i] - alpha_fitted_500_10_CO_chi_sq[i])/expt_transfer_coefficient_500_10_err[i])^2
            chi_sq_alpha_500_100[i] = ((expt_transfer_coefficient_500_100[i] - alpha_fitted_500_100_CO_chi_sq[i])/expt_transfer_coefficient_500_100_err[i])^2

            chi_sq_alpha_1000_0_1[i] = ((expt_transfer_coefficient_1000_0pt1[i] - alpha_fitted_1000_0_1_CO_chi_sq[i])/expt_transfer_coefficient_1000_0pt1_err[i])^2
            chi_sq_alpha_1000_1[i] = ((expt_transfer_coefficient_1000_1[i] - alpha_fitted_1000_1_CO_chi_sq[i])/expt_transfer_coefficient_1000_1_err[i])^2
            chi_sq_alpha_1000_10[i] = ((expt_transfer_coefficient_1000_10[i] - alpha_fitted_1000_10_CO_chi_sq[i])/expt_transfer_coefficient_1000_10_err[i])^2
            chi_sq_alpha_1000_100[i] = ((expt_transfer_coefficient_1000_100[i] - alpha_fitted_1000_100_CO_chi_sq[i])/expt_transfer_coefficient_1000_100_err[i])^2

            chi_sq_CO_order_250_0pt1_1[i] = ((expt_CO_order_250_0pt1_1[i] - CO_order_fitted_250_0pt1_1_chi_sq[i])/expt_CO_order_250_0pt1_1_err[i])^2
            chi_sq_CO_order_250_1_10[i] = ((expt_CO_order_250_1_10[i] - CO_order_fitted_250_1_10_chi_sq[i])/expt_CO_order_250_1_10_err[i])^2
            chi_sq_CO_order_250_10_100[i] = ((expt_CO_order_250_10_100[i] - CO_order_fitted_250_10_100_chi_sq[i])/expt_CO_order_250_10_100_err[i])^2

            chi_sq_CO_order_500_0pt1_1[i] = ((expt_CO_order_500_0pt1_1[i] - CO_order_fitted_500_0pt1_1_chi_sq[i])/expt_CO_order_500_0pt1_1_err[i])^2
            chi_sq_CO_order_500_1_10[i] = ((expt_CO_order_500_1_10[i] - CO_order_fitted_500_1_10_chi_sq[i])/expt_CO_order_500_1_10_err[i])^2
            chi_sq_CO_order_500_10_100[i] = ((expt_CO_order_500_10_100[i] - CO_order_fitted_500_10_100_chi_sq[i])/expt_CO_order_500_10_100_err[i])^2

            chi_sq_CO_order_1000_0pt1_1[i] = ((expt_CO_order_1000_0pt1_1[i] - CO_order_fitted_1000_0pt1_1_chi_sq[i])/expt_CO_order_1000_0pt1_1_err[i])^2
            chi_sq_CO_order_1000_1_10[i] = ((expt_CO_order_1000_1_10[i] - CO_order_fitted_1000_1_10_chi_sq[i])/expt_CO_order_1000_1_10_err[i])^2
            chi_sq_CO_order_1000_10_100[i] = ((expt_CO_order_1000_10_100[i] - CO_order_fitted_1000_10_100_chi_sq[i])/expt_CO_order_1000_10_100_err[i])^2

            chi_sq_OH_order_0pt1[i] = ((expt_OH_order_0pt1[i] - OH_order_fitted_0pt1_chi_sq[i])/expt_OH_order_0pt1_err[i])^2
            chi_sq_OH_order_1[i] = ((expt_OH_order_1[i] - OH_order_fitted_1_chi_sq[i])/expt_OH_order_1_err[i])^2
            chi_sq_OH_order_10[i] = ((expt_OH_order_10[i] - OH_order_fitted_10_chi_sq[i])/expt_OH_order_10_err[i])^2
            chi_sq_OH_order_100[i] = ((expt_OH_order_100[i] - OH_order_fitted_100_chi_sq[i])/expt_OH_order_100_err[i])^2
        end

        total_chi_sq_alpha_250_0_1 = sum(chi_sq_alpha_250_0_1)
        total_chi_sq_alpha_250_1 = sum(chi_sq_alpha_250_1)
        total_chi_sq_alpha_250_10 = sum(chi_sq_alpha_250_10)
        total_chi_sq_alpha_250_100 = sum(chi_sq_alpha_250_100)

        total_chi_sq_alpha_500_0_1 = sum(chi_sq_alpha_500_0_1)
        total_chi_sq_alpha_500_1 = sum(chi_sq_alpha_500_1)
        total_chi_sq_alpha_500_10 = sum(chi_sq_alpha_500_10)
        total_chi_sq_alpha_500_100 = sum(chi_sq_alpha_500_100)

        total_chi_sq_alpha_1000_0_1 = sum(chi_sq_alpha_1000_0_1)
        total_chi_sq_alpha_1000_1 = sum(chi_sq_alpha_1000_1)
        total_chi_sq_alpha_1000_10 = sum(chi_sq_alpha_1000_10)
        total_chi_sq_alpha_1000_100 = sum(chi_sq_alpha_1000_100)

        total_chi_sq_CO_order_250_0pt1_1 = sum(chi_sq_CO_order_250_0pt1_1)
        total_chi_sq_CO_order_250_1_10 = sum(chi_sq_CO_order_250_1_10)
        total_chi_sq_CO_order_250_10_100 = sum(chi_sq_CO_order_250_10_100)

        total_chi_sq_CO_order_500_0pt1_1 = sum(chi_sq_CO_order_500_0pt1_1)
        total_chi_sq_CO_order_500_1_10 = sum(chi_sq_CO_order_500_1_10)
        total_chi_sq_CO_order_500_10_100 = sum(chi_sq_CO_order_500_10_100)

        total_chi_sq_CO_order_1000_0pt1_1 = sum(chi_sq_CO_order_1000_0pt1_1)
        total_chi_sq_CO_order_1000_1_10 = sum(chi_sq_CO_order_1000_1_10)
        total_chi_sq_CO_order_1000_10_100 = sum(chi_sq_CO_order_1000_10_100)

        total_chi_sq_OH_order_0pt1 = sum(chi_sq_OH_order_0pt1)
        total_chi_sq_OH_order_1 = sum(chi_sq_OH_order_1)
        total_chi_sq_OH_order_10 = sum(chi_sq_OH_order_10)
        total_chi_sq_OH_order_100 = sum(chi_sq_OH_order_100)

        total_chi_sq = sum([total_chi_sq_alpha_250_0_1, total_chi_sq_alpha_250_1, total_chi_sq_alpha_250_10, total_chi_sq_alpha_250_100, total_chi_sq_alpha_500_0_1, total_chi_sq_alpha_500_1, 
                            total_chi_sq_alpha_500_10, total_chi_sq_alpha_500_100, total_chi_sq_alpha_1000_0_1, total_chi_sq_alpha_1000_1, total_chi_sq_alpha_1000_10, total_chi_sq_alpha_1000_100,
                            total_chi_sq_CO_order_250_0pt1_1, total_chi_sq_CO_order_250_1_10, total_chi_sq_CO_order_250_10_100, total_chi_sq_CO_order_500_0pt1_1, total_chi_sq_CO_order_500_1_10, 
                            total_chi_sq_CO_order_500_10_100, total_chi_sq_CO_order_1000_0pt1_1, total_chi_sq_CO_order_1000_1_10, total_chi_sq_CO_order_1000_10_100, total_chi_sq_OH_order_0pt1,
                            total_chi_sq_OH_order_1, total_chi_sq_OH_order_10, total_chi_sq_OH_order_100])
        # total_chi_sq = sum([total_chi_sq_alpha_10, total_chi_sq_CO_order, total_chi_sq_OH_order])
        return total_chi_sq
    end

    function determineHessian(obj_fun, fitted_params, perturb)
        hessian = zeros(length(fitted_params), length(fitted_params))
        for i = 1:length(fitted_params)
            for j = 1:length(fitted_params)
                x_ij = copy(fitted_params)
                x_ij[i] = x_ij[i] * (1 + perturb)
                x_ij[j] = x_ij[j] * (1 + perturb)
                f_ij = obj_fun(x_ij)

                x_i = copy(fitted_params)
                x_i[i] = x_i[i] * (1 + perturb)
                f_i = obj_fun(x_i)

                x_j = copy(fitted_params)
                x_j[j] = x_j[j] * (1 + perturb)
                f_j = obj_fun(x_j)

                f_0 = obj_fun(fitted_params)

                hessian[i, j] = (f_ij - f_i - f_j + f_0) / (perturb ^ 2)
            end
        end
        return hessian
    end

    # Run the optimization using the negative log likelihood as the objective function and get the fitted parameters and minimized neg likelihood
    basin_hopping_res = runOptimization(getNegLogLikelihood)
    res = basin_hopping_res[1]
    local_mins = basin_hopping_res[2]
    filtered_local_mins = zeros(1, length(priors)+1)
    for i = 1:global_opt_iter
        if local_mins[i,length(priors)+1] == 0
            continue
        elseif local_mins[i,length(priors)+1] == NaN
            continue
        else
            if i == 1
                filtered_local_mins[1,:] = local_mins[i,:]
            else
                filtered_local_mins = vcat(filtered_local_mins, transpose(local_mins[i,:]))
            end
        end
    end
    min_result = argmin(filtered_local_mins[:,length(priors)+1])
    fitted_params = filtered_local_mins[min_result,1:length(priors)]
    println(fitted_params)
    neg_log_likelihood_optimized = get(res, "fun", 0)
    hessian = determineHessian(getNegLogLikelihood, fitted_params, 1e-3)
    #println("The Hessian matrix")
    #println(hessian[1,:])
    #println(hessian[2,:])
    #println(hessian[3,:])
    #println(hessian[4,:])
    #println(hessian[5,:])
    total_std_dev = sum([sum(expt_transfer_coefficient_250_0pt1_err),
                         sum(expt_transfer_coefficient_250_1_err),
                         sum(expt_transfer_coefficient_250_10_err),
                         sum(expt_transfer_coefficient_250_100_err),
                         sum(expt_transfer_coefficient_500_0pt1_err),
                         sum(expt_transfer_coefficient_500_1_err),
                         sum(expt_transfer_coefficient_500_10_err),
                         sum(expt_transfer_coefficient_500_100_err),
                         sum(expt_transfer_coefficient_1000_0pt1_err),
                         sum(expt_transfer_coefficient_1000_1_err),
                         sum(expt_transfer_coefficient_1000_10_err),
                         sum(expt_transfer_coefficient_1000_100_err),    
                         sum(expt_CO_order_250_0pt1_1_err),
                         sum(expt_CO_order_250_1_10_err),
                         sum(expt_CO_order_250_10_100_err),
                         sum(expt_CO_order_500_0pt1_1_err),
                         sum(expt_CO_order_500_1_10_err),
                         sum(expt_CO_order_500_10_100_err),
                         sum(expt_CO_order_1000_0pt1_1_err),
                         sum(expt_CO_order_1000_1_10_err),
                         sum(expt_CO_order_1000_10_100_err),
                         sum(expt_OH_order_0pt1_err),
                         sum(expt_OH_order_1_err),
                         sum(expt_OH_order_10_err),
                         sum(expt_OH_order_100_err)])
    
    total_num_data = sum([length(expt_transfer_coefficient_250_0pt1_err),
                          length(expt_transfer_coefficient_250_1_err),
                          length(expt_transfer_coefficient_250_10_err),
                          length(expt_transfer_coefficient_250_100_err),
                          length(expt_transfer_coefficient_500_0pt1_err),
                          length(expt_transfer_coefficient_500_1_err),
                          length(expt_transfer_coefficient_500_10_err),
                          length(expt_transfer_coefficient_500_100_err),
                          length(expt_transfer_coefficient_1000_0pt1_err),
                          length(expt_transfer_coefficient_1000_1_err),
                          length(expt_transfer_coefficient_1000_10_err),
                          length(expt_transfer_coefficient_1000_100_err),    
                          length(expt_CO_order_250_0pt1_1_err),
                          length(expt_CO_order_250_1_10_err),
                          length(expt_CO_order_250_10_100_err),
                          length(expt_CO_order_500_0pt1_1_err),
                          length(expt_CO_order_500_1_10_err),
                          length(expt_CO_order_500_10_100_err),
                          length(expt_CO_order_1000_0pt1_1_err),
                          length(expt_CO_order_1000_1_10_err),
                          length(expt_CO_order_1000_10_100_err),
                          length(expt_OH_order_0pt1_err),
                          length(expt_OH_order_1_err),
                          length(expt_OH_order_10_err),
                          length(expt_OH_order_100_err)])

    total_avg_std_dev = total_std_dev / total_num_data
    total_avg_var = total_avg_std_dev ^ 2
    #covariance_mat_new = (2 * total_avg_var) .* inv(1/total_num_data .* hessian)
    #covariance_mat_new = inv(1/total_num_data .* hessian)
    std_error = zeros(length(fitted_params), 1)
    if isapprox(det(BigFloat.(hessian)), 0, atol = 1e-18) == true
        println("Hessian is non-invertible")
        covariance_mat_new = hessian
    else
        covariance_mat_new = inv(hessian)
        println("Using finite difference Hessian")
        for i = 1:length(fitted_params)
            @printf("Approximate Standard Error of parameter %s is: %.3f ", i, np.sqrt(covariance_mat_new[i,i]))
            std_error[i,1] = np.sqrt(covariance_mat_new[i,i])
            println()
        end
    end
    #println("The Covariance matrix")
    #println(covariance_mat_new[1,:])
    #println(covariance_mat_new[2,:])
    #println(covariance_mat_new[3,:])
    #println(covariance_mat_new[4,:])
    #println(covariance_mat_new[5,:])
    
    # Change directory for storing fitted parameters. They all go into the same spreadsheet.
    cd("..\\Ag90Pd10")

    # Simulation with fitted (i.e. neg log likelikood optimized) parameters. Figure is made inside the following function.
    # fitted_params = [-0.8, 0.65, 0.5, 0.041, 0.8, 0.2, 0.2, 0.01]
    fitted_data = simulateWithFittedParams(fitted_params)
    t_CO_CO_order = fitted_data[1]
    t_CO_0_1_250_CO_order = t_CO_CO_order[1][1,:]
    t_CO_1_250_CO_order = t_CO_CO_order[1][2,:]
    t_CO_10_250_CO_order = t_CO_CO_order[1][3,:]
    t_CO_100_250_CO_order = t_CO_CO_order[1][4,:]
    t_CO_0_1_500_CO_order = t_CO_CO_order[2][1,:]
    t_CO_1_500_CO_order = t_CO_CO_order[2][2,:]
    t_CO_10_500_CO_order = t_CO_CO_order[2][3,:]
    t_CO_100_500_CO_order = t_CO_CO_order[2][4,:]
    t_CO_0_1_1000_CO_order = t_CO_CO_order[3][1,:]
    t_CO_1_1000_CO_order = t_CO_CO_order[3][2,:]
    t_CO_10_1000_CO_order = t_CO_CO_order[3][3,:]
    t_CO_100_1000_CO_order = t_CO_CO_order[3][4,:]
    t_CO_OH_order = fitted_data[2]
    t_CO_0_1_250_OH_order = t_CO_OH_order[1][1,:]
    t_CO_0_1_500_OH_order = t_CO_OH_order[1][2,:]
    t_CO_0_1_1000_OH_order = t_CO_OH_order[1][3,:]
    t_CO_1_250_OH_order = t_CO_OH_order[2][1,:]
    t_CO_1_500_OH_order = t_CO_OH_order[2][2,:]
    t_CO_1_1000_OH_order = t_CO_OH_order[2][3,:]
    t_CO_10_250_OH_order = t_CO_OH_order[3][1,:]
    t_CO_10_500_OH_order = t_CO_OH_order[3][2,:]
    t_CO_10_1000_OH_order = t_CO_OH_order[3][3,:]
    t_CO_100_250_OH_order = t_CO_OH_order[4][1,:]
    t_CO_100_500_OH_order = t_CO_OH_order[4][2,:]
    t_CO_100_1000_OH_order = t_CO_OH_order[4][3,:]
    t_OH_CO_order = fitted_data[3]
    t_OH_0_1_250_CO_order = t_OH_CO_order[1][1,:]
    t_OH_1_250_CO_order = t_OH_CO_order[1][2,:]
    t_OH_10_250_CO_order = t_OH_CO_order[1][3,:]
    t_OH_100_250_CO_order = t_OH_CO_order[1][4,:]
    t_OH_0_1_500_CO_order = t_OH_CO_order[2][1,:]
    t_OH_1_500_CO_order = t_OH_CO_order[2][2,:]
    t_OH_10_500_CO_order = t_OH_CO_order[2][3,:]
    t_OH_100_500_CO_order = t_OH_CO_order[2][4,:]
    t_OH_0_1_1000_CO_order = t_OH_CO_order[3][1,:]
    t_OH_1_1000_CO_order = t_OH_CO_order[3][2,:]
    t_OH_10_1000_CO_order = t_OH_CO_order[3][3,:]
    t_OH_100_1000_CO_order = t_OH_CO_order[3][4,:]
    t_OH_OH_order = fitted_data[4]
    t_OH_0_1_250_OH_order = t_OH_OH_order[1][1,:]
    t_OH_0_1_500_OH_order = t_OH_OH_order[1][2,:]
    t_OH_0_1_1000_OH_order = t_OH_OH_order[1][3,:]
    t_OH_1_250_OH_order = t_OH_OH_order[2][1,:]
    t_OH_1_500_OH_order = t_OH_OH_order[2][2,:]
    t_OH_1_1000_OH_order = t_OH_OH_order[2][3,:]
    t_OH_10_250_OH_order = t_OH_OH_order[3][1,:]
    t_OH_10_500_OH_order = t_OH_OH_order[3][2,:]
    t_OH_10_1000_OH_order = t_OH_OH_order[3][3,:]
    t_OH_100_250_OH_order = t_OH_OH_order[4][1,:]
    t_OH_100_500_OH_order = t_OH_OH_order[4][2,:]
    t_OH_100_1000_OH_order = t_OH_OH_order[4][3,:]
    alpha_250_fitted = fitted_data[5]
    alpha_250_0pt1_fitted = alpha_250_fitted[1,:]
    alpha_250_1_fitted = alpha_250_fitted[2,:]
    alpha_250_10_fitted = alpha_250_fitted[3,:]
    alpha_250_100_fitted = alpha_250_fitted[4,:]
    alpha_500_fitted = fitted_data[6]
    alpha_500_0pt1_fitted = alpha_500_fitted[1,:]
    alpha_500_1_fitted = alpha_500_fitted[2,:]
    alpha_500_10_fitted = alpha_500_fitted[3,:]
    alpha_500_100_fitted = alpha_500_fitted[4,:]
    alpha_1000_fitted = fitted_data[7]
    alpha_1000_0pt1_fitted = alpha_1000_fitted[1,:]
    alpha_1000_1_fitted = alpha_1000_fitted[2,:]
    alpha_1000_10_fitted = alpha_1000_fitted[3,:]
    alpha_1000_100_fitted = alpha_1000_fitted[4,:]
    CO_order_250_fitted = fitted_data[8]
    CO_order_250_0pt1_1_fitted = CO_order_250_fitted[1,:]
    CO_order_250_1_10_fitted = CO_order_250_fitted[2,:]
    CO_order_250_10_100_fitted = CO_order_250_fitted[3,:]
    CO_order_500_fitted = fitted_data[9]
    CO_order_500_0pt1_1_fitted = CO_order_500_fitted[1,:]
    CO_order_500_1_10_fitted = CO_order_500_fitted[2,:]
    CO_order_500_10_100_fitted = CO_order_500_fitted[3,:]
    CO_order_1000_fitted = fitted_data[10]
    CO_order_1000_0pt1_1_fitted = CO_order_1000_fitted[1,:]
    CO_order_1000_1_10_fitted = CO_order_1000_fitted[2,:]
    CO_order_1000_10_100_fitted = CO_order_1000_fitted[3,:]
    OH_order_fitted = fitted_data[11]
    OH_order_0pt1_fitted = OH_order_fitted[1,:]
    OH_order_1_fitted = OH_order_fitted[2,:]
    OH_order_10_fitted = OH_order_fitted[3,:]
    OH_order_100_fitted = OH_order_fitted[4,:]
    total_rate_CO_order = fitted_data[12]
    total_rate_0_1_250_CO_order = total_rate_CO_order[1][1,:]
    total_rate_1_250_CO_order = total_rate_CO_order[1][2,:]
    total_rate_10_250_CO_order = total_rate_CO_order[1][3,:]
    total_rate_100_250_CO_order = total_rate_CO_order[1][4,:]
    total_rate_0_1_500_CO_order = total_rate_CO_order[2][1,:]
    total_rate_1_500_CO_order = total_rate_CO_order[2][2,:]
    total_rate_10_500_CO_order = total_rate_CO_order[2][3,:]
    total_rate_100_500_CO_order = total_rate_CO_order[2][4,:]
    total_rate_0_1_1000_CO_order = total_rate_CO_order[3][1,:]
    total_rate_1_1000_CO_order = total_rate_CO_order[3][2,:]
    total_rate_10_1000_CO_order = total_rate_CO_order[3][3,:]
    total_rate_100_1000_CO_order = total_rate_CO_order[3][4,:]
    total_rate_OH_order = fitted_data[13]
    total_rate_0_1_250_OH_order = total_rate_OH_order[1][1,:]
    total_rate_0_1_500_OH_order = total_rate_OH_order[1][2,:]
    total_rate_0_1_1000_OH_order = total_rate_OH_order[1][3,:]
    total_rate_1_250_OH_order = total_rate_OH_order[2][1,:]
    total_rate_1_500_OH_order = total_rate_OH_order[2][2,:]
    total_rate_1_1000_OH_order = total_rate_OH_order[2][3,:]
    total_rate_10_250_OH_order = total_rate_OH_order[3][1,:]
    total_rate_10_500_OH_order = total_rate_OH_order[3][2,:]
    total_rate_10_1000_OH_order = total_rate_OH_order[3][3,:]
    total_rate_100_250_OH_order = total_rate_OH_order[4][1,:]
    total_rate_100_500_OH_order = total_rate_OH_order[4][2,:]
    total_rate_100_1000_OH_order = total_rate_OH_order[4][3,:]
    total_chi_sq = getChiSquared(fitted_params)
    reduced_chi_sq = total_chi_sq / (total_num_data - length(fitted_params))
    @printf("Reduced Chi^2 is: %.3f", total_chi_sq / (total_num_data - length(fitted_params)))
    println()
    XLSX.openxlsx("Ag90Pd10_single_site_LH_CO_differential_with_lat_int_coverages_rates.xlsx", mode = "rw") do f
        ### Write all of the coverages and rates here
        sheet = f[1]
        sheet["A1"] = "Model potential (V vs RHE)"
        sheet["A2",dim=1] = E_array_simulated
        sheet["B1"] = "Experimental potential (V vs RHE)"
        sheet["B2",dim=1] = expt_rhe_potential
        sheet["C1"] = "Experimental potential (V vs SHE)"
        sheet["C2",dim=1] = expt_she_potential
        sheet["D1"] = "Theta_CO at 0.1% CO, 0.25 M KOH - CO order"
        sheet["D2",dim=1] = t_CO_0_1_250_CO_order
        sheet["E1"] = "Theta_CO at 1% CO, 0.25 M KOH - CO order"
        sheet["E2",dim=1] = t_CO_1_250_CO_order
        sheet["F1"] = "Theta_CO at 10% CO, 0.25 M KOH - CO order"
        sheet["F2",dim=1] = t_CO_10_250_CO_order
        sheet["G1"] = "Theta_CO at 100% CO, 0.25 M KOH - CO order"
        sheet["G2",dim=1] = t_CO_100_250_CO_order
        sheet["H1"] = "Theta_CO at 0.1% CO, 0.5 M KOH - CO order"
        sheet["H2",dim=1] = t_CO_0_1_500_CO_order
        sheet["I1"] = "Theta_CO at 1% CO, 0.5 M KOH - CO order"
        sheet["I2",dim=1] = t_CO_1_500_CO_order
        sheet["J1"] = "Theta_CO at 10% CO, 0.5 M KOH - CO order"
        sheet["J2",dim=1] = t_CO_10_500_CO_order
        sheet["K1"] = "Theta_CO at 100% CO, 0.5 M KOH - CO order"
        sheet["K2",dim=1] = t_CO_100_500_CO_order
        sheet["L1"] = "Theta_CO at 0.1% CO, 1 M KOH - CO order"
        sheet["L2",dim=1] = t_CO_0_1_1000_CO_order
        sheet["M1"] = "Theta_CO at 1% CO, 1 M KOH - CO order"
        sheet["M2",dim=1] = t_CO_1_1000_CO_order
        sheet["N1"] = "Theta_CO at 10% CO, 1 M KOH - CO order"
        sheet["N2",dim=1] = t_CO_10_1000_CO_order
        sheet["O1"] = "Theta_CO at 100% CO, 1 M KOH - CO order"
        sheet["O2",dim=1] = t_CO_100_1000_CO_order
        sheet["P1"] = "Theta_OH at 0.1% CO, 0.25 M KOH - CO order"
        sheet["P2",dim=1] = t_OH_0_1_250_CO_order
        sheet["Q1"] = "Theta_OH at 1% CO, 0.25 M KOH - CO order"
        sheet["Q2",dim=1] = t_OH_1_250_CO_order
        sheet["R1"] = "Theta_OH at 10% CO, 0.25 M KOH - CO order"
        sheet["R2",dim=1] = t_OH_10_250_CO_order
        sheet["S1"] = "Theta_OH at 100% CO, 0.25 M KOH - CO order"
        sheet["S2",dim=1] = t_OH_100_250_CO_order
        sheet["T1"] = "Theta_OH at 0.1% CO, 0.5 M KOH - CO order"
        sheet["T2",dim=1] = t_OH_0_1_500_CO_order
        sheet["U1"] = "Theta_OH at 1% CO, 0.5 M KOH - CO order"
        sheet["U2",dim=1] = t_OH_1_500_CO_order
        sheet["V1"] = "Theta_OH at 10% CO, 0.5 M KOH - CO order"
        sheet["V2",dim=1] = t_OH_10_500_CO_order
        sheet["W1"] = "Theta_OH at 100% CO, 0.5 M KOH - CO order"
        sheet["W2",dim=1] = t_OH_100_500_CO_order
        sheet["X1"] = "Theta_OH at 0.1% CO, 1 M KOH - CO order"
        sheet["X2",dim=1] = t_OH_0_1_1000_CO_order
        sheet["Y1"] = "Theta_OH at 1% CO, 1 M KOH - CO order"
        sheet["Y2",dim=1] = t_OH_1_1000_CO_order
        sheet["Z1"] = "Theta_OH at 10% CO, 1 M KOH - CO order"
        sheet["Z2",dim=1] = t_OH_10_1000_CO_order
        sheet["AA1"] = "Theta_OH at 100% CO, 1 M KOH - CO order"
        sheet["AA2",dim=1] = t_OH_100_1000_CO_order
        sheet["AB1"] = "Total Rate at 0.1% CO, 0.25 M KOH - CO order"
        sheet["AB2",dim=1] = total_rate_0_1_250_CO_order
        sheet["AC1"] = "Total Rate at 1% CO, 0.25 M KOH - CO order"
        sheet["AC2",dim=1] = total_rate_1_250_CO_order
        sheet["AD1"] = "Total Rate at 10% CO, 0.25 M KOH - CO order"
        sheet["AD2",dim=1] = total_rate_10_250_CO_order
        sheet["AE1"] = "Total Rate at 100% CO, 0.25 M KOH - CO order"
        sheet["AE2",dim=1] = total_rate_100_250_CO_order
        sheet["AF1"] = "Total Rate at 0.1% CO, 0.5 M KOH - CO order"
        sheet["AF2",dim=1] = total_rate_0_1_500_CO_order
        sheet["AG1"] = "Total Rate at 1% CO, 0.5 M KOH - CO order"
        sheet["AG2",dim=1] = total_rate_1_500_CO_order
        sheet["AH1"] = "Total Rate at 10% CO, 0.5 M KOH - CO order"
        sheet["AH2",dim=1] = total_rate_10_500_CO_order
        sheet["AI1"] = "Total Rate at 100% CO, 0.5 M KOH - CO order"
        sheet["AI2",dim=1] = total_rate_100_500_CO_order
        sheet["AJ1"] = "Total Rate at 0.1% CO, 1 M KOH - CO order"
        sheet["AJ2",dim=1] = total_rate_0_1_1000_CO_order
        sheet["AK1"] = "Total Rate at 1% CO, 1 M KOH - CO order"
        sheet["AK2",dim=1] = total_rate_1_1000_CO_order
        sheet["AL1"] = "Total Rate at 10% CO, 1 M KOH - CO order"
        sheet["AL2",dim=1] = total_rate_10_1000_CO_order
        sheet["AM1"] = "Total Rate at 100% CO, 1 M KOH - CO order"
        sheet["AM2",dim=1] = total_rate_100_1000_CO_order
        sheet["AN1"] = "Theta_CO at 0.1% CO, 0.25 M KOH - OH order"
        sheet["AN2",dim=1] = t_CO_0_1_250_OH_order
        sheet["AO1"] = "Theta_CO at 0.1% CO, 0.5 M KOH - OH order"
        sheet["AO2",dim=1] = t_CO_0_1_500_OH_order
        sheet["AP1"] = "Theta_CO at 0.1% CO, 1 M KOH - OH order"
        sheet["AP2",dim=1] = t_CO_0_1_1000_OH_order
        sheet["AQ1"] = "Theta_CO at 1% CO, 0.25 M KOH - OH order"
        sheet["AQ2",dim=1] = t_CO_1_250_OH_order
        sheet["AR1"] = "Theta_CO at 1% CO, 0.5 M KOH - OH order"
        sheet["AR2",dim=1] = t_CO_1_500_OH_order
        sheet["AS1"] = "Theta_CO at 1% CO, 1 M KOH - OH order"
        sheet["AS2",dim=1] = t_CO_1_1000_OH_order
        sheet["AT1"] = "Theta_CO at 10% CO, 0.25 M KOH - OH order"
        sheet["AT2",dim=1] = t_CO_10_250_OH_order
        sheet["AU1"] = "Theta_CO at 10% CO, 0.5 M KOH - OH order"
        sheet["AU2",dim=1] = t_CO_10_500_OH_order
        sheet["AV1"] = "Theta_CO at 10% CO, 1 M KOH - OH order"
        sheet["AV2",dim=1] = t_CO_10_1000_OH_order
        sheet["AW1"] = "Theta_CO at 100% CO, 0.25 M KOH - OH order"
        sheet["AW2",dim=1] = t_CO_100_250_OH_order
        sheet["AX1"] = "Theta_CO at 100% CO, 0.5 M KOH - OH order"
        sheet["AX2",dim=1] = t_CO_100_500_OH_order
        sheet["AY1"] = "Theta_CO at 100% CO, 1 M KOH - OH order"
        sheet["AY2",dim=1] = t_CO_100_1000_OH_order
        sheet["AZ1"] = "Theta_OH at 0.1% CO, 0.25 M KOH - OH order"
        sheet["AZ2",dim=1] = t_OH_0_1_250_OH_order
        sheet["BA1"] = "Theta_OH at 0.1% CO, 0.5 M KOH - OH order"
        sheet["BA2",dim=1] = t_OH_0_1_500_OH_order
        sheet["BB1"] = "Theta_OH at 0.1% CO, 1 M KOH - OH order"
        sheet["BB2",dim=1] = t_OH_0_1_1000_OH_order
        sheet["BC1"] = "Theta_OH at 1% CO, 0.25 M KOH - OH order"
        sheet["BC2",dim=1] = t_OH_1_250_OH_order
        sheet["BD1"] = "Theta_OH at 1% CO, 0.5 M KOH - OH order"
        sheet["BD2",dim=1] = t_OH_1_500_OH_order
        sheet["BE1"] = "Theta_OH at 1% CO, 1 M KOH - OH order"
        sheet["BE2",dim=1] = t_OH_1_1000_OH_order
        sheet["BF1"] = "Theta_OH at 10% CO, 0.25 M KOH - OH order"
        sheet["BF2",dim=1] = t_OH_10_250_OH_order
        sheet["BG1"] = "Theta_OH at 10% CO, 0.5 M KOH - OH order"
        sheet["BG2",dim=1] = t_OH_10_500_OH_order
        sheet["BH1"] = "Theta_OH at 10% CO, 1 M KOH - OH order"
        sheet["BH2",dim=1] = t_OH_10_1000_OH_order
        sheet["BI1"] = "Theta_OH at 100% CO, 0.25 M KOH - OH order"
        sheet["BI2",dim=1] = t_OH_100_250_OH_order
        sheet["BJ1"] = "Theta_OH at 100% CO, 0.5 M KOH - OH order"
        sheet["BJ2",dim=1] = t_OH_100_500_OH_order
        sheet["BK1"] = "Theta_OH at 100% CO, 1 M KOH - OH order"
        sheet["BK2",dim=1] = t_OH_100_1000_OH_order
        sheet["BL1"] = "Total Rate at 0.1% CO, 0.25 M KOH - OH order"
        sheet["BL2",dim=1] = total_rate_0_1_250_OH_order
        sheet["BM1"] = "Total Rate at 0.1% CO, 0.5 M KOH - OH order"
        sheet["BM2",dim=1] = total_rate_0_1_500_OH_order
        sheet["BN1"] = "Total Rate at 0.1% CO, 1 M KOH - OH order"
        sheet["BN2",dim=1] = total_rate_0_1_1000_OH_order
        sheet["BO1"] = "Total Rate at 1% CO, 0.25 M KOH - OH order"
        sheet["BO2",dim=1] = total_rate_1_250_OH_order
        sheet["BP1"] = "Total Rate at 1% CO, 0.5 M KOH - OH order"
        sheet["BP2",dim=1] = total_rate_1_500_OH_order
        sheet["BQ1"] = "Total Rate at 1% CO, 1 M KOH - OH order"
        sheet["BQ2",dim=1] = total_rate_1_1000_OH_order
        sheet["BR1"] = "Total Rate at 10% CO, 0.25 M KOH - OH order"
        sheet["BR2",dim=1] = total_rate_10_250_OH_order
        sheet["BS1"] = "Total Rate at 10% CO, 0.5 M KOH - OH order"
        sheet["BS2",dim=1] = total_rate_10_500_OH_order
        sheet["BT1"] = "Total Rate at 10% CO, 1 M KOH - OH order"
        sheet["BT2",dim=1] = total_rate_10_1000_OH_order
        sheet["BU1"] = "Total Rate at 100% CO, 0.25 M KOH - OH order"
        sheet["BU2",dim=1] = total_rate_100_250_OH_order
        sheet["BV1"] = "Total Rate at 100% CO, 0.5 M KOH - OH order"
        sheet["BV2",dim=1] = total_rate_100_500_OH_order
        sheet["BW1"] = "Total Rate at 100% CO, 1 M KOH - OH order"
        sheet["BW2",dim=1] = total_rate_100_1000_OH_order
        sheet["BX1"] = "Model potential (V vs SHE)"
        sheet["BX2",dim=1] = E_array_OH_simulated
    end
    #### Write all of the observables and errors here, as well as the fitted params and covariance etc
    XLSX.openxlsx("Ag90Pd10_single_site_LH_CO_differential_with_lat_int_observables.xlsx", mode = "rw") do f
        sheet = f[1]
        sheet["A1"] = "Model potential (V vs RHE)"
        sheet["A2",dim=1] = E_array_simulated
        sheet["B1"] = "Experimental potential (V vs RHE)"
        sheet["B2",dim=1] = expt_rhe_potential
        sheet["C1"] = "Experimental potential (V vs SHE)"
        sheet["C2",dim=1] = expt_she_potential
        sheet["D1"] = "Model transfer coefficient at 0.1% CO, 0.25 M KOH"
        sheet["D2",dim=1] = alpha_250_0pt1_fitted
        sheet["E1"] = "Experimental transfer coefficient at 0.1% CO, 0.25 M KOH"
        sheet["E2",dim=1] = expt_transfer_coefficient_250_0pt1
        sheet["F1"] = "Experimental transfer coefficient error at 0.1% CO, 0.25 M KOH"
        sheet["F2",dim=1] = expt_transfer_coefficient_250_0pt1_err
        sheet["G1"] = "Model transfer coefficient at 1% CO, 0.25 M KOH"
        sheet["G2",dim=1] = alpha_250_1_fitted
        sheet["H1"] = "Experimental transfer coefficient at 1% CO, 0.25 M KOH"
        sheet["H2",dim=1] = expt_transfer_coefficient_250_1
        sheet["I1"] = "Experimental transfer coefficient error at 1% CO, 0.25 M KOH"
        sheet["I2",dim=1] = expt_transfer_coefficient_250_1_err
        sheet["J1"] = "Model transfer coefficient at 10% CO, 0.25 M KOH"
        sheet["J2",dim=1] = alpha_250_10_fitted
        sheet["K1"] = "Experimental transfer coefficient at 10% CO, 0.25 M KOH"
        sheet["K2",dim=1] = expt_transfer_coefficient_250_10
        sheet["L1"] = "Experimental transfer coefficient error at 10% CO, 0.25 M KOH"
        sheet["L2",dim=1] = expt_transfer_coefficient_250_10_err
        sheet["M1"] = "Model transfer coefficient at 100% CO, 0.25 M KOH"
        sheet["M2",dim=1] = alpha_250_100_fitted
        sheet["N1"] = "Experimental transfer coefficient at 100% CO, 0.25 M KOH"
        sheet["N2",dim=1] = expt_transfer_coefficient_250_100
        sheet["O1"] = "Experimental transfer coefficient error at 100% CO, 0.25 M KOH"
        sheet["O2",dim=1] = expt_transfer_coefficient_250_100_err
        sheet["P1"] = "Model transfer coefficient at 0.1% CO, 0.5 M KOH"
        sheet["P2",dim=1] = alpha_500_0pt1_fitted
        sheet["Q1"] = "Experimental transfer coefficient at 0.1% CO, 0.5 M KOH"
        sheet["Q2",dim=1] = expt_transfer_coefficient_500_0pt1
        sheet["R1"] = "Experimental transfer coefficient error at 0.1% CO, 0.5 M KOH"
        sheet["R2",dim=1] = expt_transfer_coefficient_500_0pt1_err
        sheet["S1"] = "Model transfer coefficient at 1% CO, 0.5 M KOH"
        sheet["S2",dim=1] = alpha_500_1_fitted
        sheet["T1"] = "Experimental transfer coefficient at 1% CO, 0.5 M KOH"
        sheet["T2",dim=1] = expt_transfer_coefficient_500_1
        sheet["U1"] = "Experimental transfer coefficient error at 1% CO, 0.5 M KOH"
        sheet["U2",dim=1] = expt_transfer_coefficient_500_1_err
        sheet["V1"] = "Model transfer coefficient at 10% CO, 0.5 M KOH"
        sheet["V2",dim=1] = alpha_500_10_fitted
        sheet["W1"] = "Experimental transfer coefficient at 10% CO, 0.5 M KOH"
        sheet["W2",dim=1] = expt_transfer_coefficient_500_10
        sheet["X1"] = "Experimental transfer coefficient error at 10% CO, 0.5 M KOH"
        sheet["X2",dim=1] = expt_transfer_coefficient_500_10_err
        sheet["Y1"] = "Model transfer coefficient at 100% CO, 0.5 M KOH"
        sheet["Y2",dim=1] = alpha_500_100_fitted
        sheet["Z1"] = "Experimental transfer coefficient at 100% CO, 0.5 M KOH"
        sheet["Z2",dim=1] = expt_transfer_coefficient_500_100
        sheet["AA1"] = "Experimental transfer coefficient error at 100% CO, 0.5 M KOH"
        sheet["AA2",dim=1] = expt_transfer_coefficient_500_100_err
        sheet["AB1"] = "Model transfer coefficient at 0.1% CO, 1 M KOH"
        sheet["AB2",dim=1] = alpha_1000_0pt1_fitted
        sheet["AC1"] = "Experimental transfer coefficient at 0.1% CO, 1 M KOH"
        sheet["AC2",dim=1] = expt_transfer_coefficient_1000_0pt1
        sheet["AD1"] = "Experimental transfer coefficient error at 0.1% CO, 1 M KOH"
        sheet["AD2",dim=1] = expt_transfer_coefficient_1000_0pt1_err
        sheet["AE1"] = "Model transfer coefficient at 1% CO, 1 M KOH"
        sheet["AE2",dim=1] = alpha_1000_1_fitted
        sheet["AF1"] = "Experimental transfer coefficient at 1% CO, 1 M KOH"
        sheet["AF2",dim=1] = expt_transfer_coefficient_1000_1
        sheet["AG1"] = "Experimental transfer coefficient error at 1% CO, 1 M KOH"
        sheet["AG2",dim=1] = expt_transfer_coefficient_1000_1_err
        sheet["AH1"] = "Model transfer coefficient at 10% CO, 1 M KOH"
        sheet["AH2",dim=1] = alpha_1000_10_fitted
        sheet["AI1"] = "Experimental transfer coefficient at 10% CO, 1 M KOH"
        sheet["AI2",dim=1] = expt_transfer_coefficient_1000_10
        sheet["AJ1"] = "Experimental transfer coefficient error at 10% CO, 1 M KOH"
        sheet["AJ2",dim=1] = expt_transfer_coefficient_1000_10_err
        sheet["AK1"] = "Model transfer coefficient at 100% CO, 1 M KOH"
        sheet["AK2",dim=1] = alpha_1000_100_fitted
        sheet["AL1"] = "Experimental transfer coefficient at 100% CO, 1 M KOH"
        sheet["AL2",dim=1] = expt_transfer_coefficient_1000_100
        sheet["AM1"] = "Experimental transfer coefficient error at 100% CO, 1 M KOH"
        sheet["AM2",dim=1] = expt_transfer_coefficient_1000_100_err
        sheet["AN1"] = "Model CO order at 0.25 M KOH, 0.1% - 1% CO"
        sheet["AN2",dim=1] = CO_order_250_0pt1_1_fitted
        sheet["AO1"] = "Experimental CO order at 0.25 M KOH, 0.1% - 1% CO"
        sheet["AO2",dim=1] = expt_CO_order_250_0pt1_1
        sheet["AP1"] = "Experimental CO order error at 0.25 M KOH, 0.1% - 1% CO"
        sheet["AP2",dim=1] = expt_CO_order_250_0pt1_1_err
        sheet["AQ1"] = "Model CO order at 0.25 M KOH, 1% - 10% CO"
        sheet["AQ2",dim=1] = CO_order_250_1_10_fitted
        sheet["AR1"] = "Experimental CO order at 0.25 M KOH, 1% - 10% CO"
        sheet["AR2",dim=1] = expt_CO_order_250_1_10
        sheet["AS1"] = "Experimental CO order error at 0.25 M KOH, 1% - 10% CO"
        sheet["AS2",dim=1] = expt_CO_order_250_1_10_err
        sheet["AT1"] = "Model CO order at 0.25 M KOH, 10% - 100% CO"
        sheet["AT2",dim=1] = CO_order_250_10_100_fitted
        sheet["AU1"] = "Experimental CO order at 0.25 M KOH, 10% - 100% CO"
        sheet["AU2",dim=1] = expt_CO_order_250_10_100
        sheet["AV1"] = "Experimental CO order error at 0.25 M KOH, 10% - 100% CO"
        sheet["AV2",dim=1] = expt_CO_order_250_10_100_err
        sheet["AW1"] = "Model CO order at 0.5 M KOH, 0.1% - 1% CO"
        sheet["AW2",dim=1] = CO_order_500_0pt1_1_fitted
        sheet["AX1"] = "Experimental CO order at 0.5 M KOH, 0.1% - 1% CO"
        sheet["AX2",dim=1] = expt_CO_order_500_0pt1_1
        sheet["AY1"] = "Experimental CO order error at 0.5 M KOH, 0.1% - 1% CO"
        sheet["AY2",dim=1] = expt_CO_order_500_0pt1_1_err
        sheet["AZ1"] = "Model CO order at 0.5 M KOH, 1% - 10% CO"
        sheet["AZ2",dim=1] = CO_order_500_1_10_fitted
        sheet["BA1"] = "Experimental CO order at 0.5 M KOH, 1% - 10% CO"
        sheet["BA2",dim=1] = expt_CO_order_500_1_10
        sheet["BB1"] = "Experimental CO order error at 0.5 M KOH, 1% - 10% CO"
        sheet["BB2",dim=1] = expt_CO_order_500_1_10_err
        sheet["BC1"] = "Model CO order at 0.5 M KOH, 10% - 100% CO"
        sheet["BC2",dim=1] = CO_order_500_10_100_fitted
        sheet["BD1"] = "Experimental CO order at 0.5 M KOH, 10% - 100% CO"
        sheet["BD2",dim=1] = expt_CO_order_500_10_100
        sheet["BE1"] = "Experimental CO order error at 0.5 M KOH, 10% - 100% CO"
        sheet["BE2",dim=1] = expt_CO_order_500_10_100_err
        sheet["BF1"] = "Model CO order at 1 M KOH, 0.1% - 1% CO"
        sheet["BF2",dim=1] = CO_order_1000_0pt1_1_fitted
        sheet["BG1"] = "Experimental CO order at 1 M KOH, 0.1% - 1% CO"
        sheet["BG2",dim=1] = expt_CO_order_1000_0pt1_1
        sheet["BH1"] = "Experimental CO order error at 1 M KOH, 0.1% - 1% CO"
        sheet["BH2",dim=1] = expt_CO_order_1000_0pt1_1_err
        sheet["BI1"] = "Model CO order at 1 M KOH, 1% - 10% CO"
        sheet["BI2",dim=1] = CO_order_1000_1_10_fitted
        sheet["BJ1"] = "Experimental CO order at 1 M KOH, 1% - 10% CO"
        sheet["BJ2",dim=1] = expt_CO_order_1000_1_10
        sheet["BK1"] = "Experimental CO order error at 1 M KOH, 1% - 10% CO"
        sheet["BK2",dim=1] = expt_CO_order_1000_1_10_err
        sheet["BL1"] = "Model CO order at 1 M KOH, 10% - 100% CO"
        sheet["BL2",dim=1] = CO_order_1000_10_100_fitted
        sheet["BM1"] = "Experimental CO order at 1 M KOH, 10% - 100% CO"
        sheet["BM2",dim=1] = expt_CO_order_1000_10_100
        sheet["BN1"] = "Experimental CO order error at 1 M KOH, 10% - 100% CO"
        sheet["BN2",dim=1] = expt_CO_order_1000_10_100_err
        sheet["BO1"] = "Model OH order at 0.1% CO"
        sheet["BO2",dim=1] = OH_order_0pt1_fitted
        sheet["BP1"] = "Experimental OH order at 0.1% CO"
        sheet["BP2",dim=1] = expt_OH_order_0pt1
        sheet["BQ1"] = "Experimental OH order error at 0.1% CO"
        sheet["BQ2",dim=1] = expt_OH_order_0pt1_err
        sheet["BR1"] = "Model OH order at 1% CO"
        sheet["BR2",dim=1] = OH_order_1_fitted
        sheet["BS1"] = "Experimental OH order at 1% CO"
        sheet["BS2",dim=1] = expt_OH_order_1
        sheet["BT1"] = "Experimental OH order error at 1% CO"
        sheet["BT2",dim=1] = expt_OH_order_1_err
        sheet["BU1"] = "Model OH order at 10% CO"
        sheet["BU2",dim=1] = OH_order_10_fitted
        sheet["BV1"] = "Experimental OH order at 10% CO"
        sheet["BV2",dim=1] = expt_OH_order_10
        sheet["BW1"] = "Experimental OH order error at 10% CO"
        sheet["BW2",dim=1] = expt_OH_order_10_err
        sheet["BX1"] = "Model OH order at 100% CO"
        sheet["BX2",dim=1] = OH_order_100_fitted
        sheet["BY1"] = "Experimental OH order at 100% CO"
        sheet["BY2",dim=1] = expt_OH_order_100
        sheet["BZ1"] = "Experimental OH order error at 100% CO"
        sheet["BZ2",dim=1] = expt_OH_order_100_err
        sheet["CA1"] = "Total Chi Squared"
        sheet["CA2"] = total_chi_sq
        sheet["CA3"] = "Reduced Chi Squared"
        sheet["CA4"] = reduced_chi_sq
        sheet["CB1"] = "Covariance Matrix from Finite Difference"
        sheet["CB2"] = covariance_mat_new
        sheet["CH1"] = "Parameter fits"
        sheet["CH2",dim=1] = fitted_params
        sheet["CI1"] = "Std Error from Finite Difference"
        sheet["CI2"] = std_error
        sheet["CJ1"] = "Model potential (V vs SHE)"
        sheet["CJ2",dim=1] = E_array_OH_simulated
    end
    ### Write the optimization trajectory here
    XLSX.openxlsx("Ag90Pd10_single_site_LH_CO_differential_with_lat_int_local_minima.xlsx", mode = "rw") do f
        sheet = f[1]
        sheet["A1"] = "dG_CO"
        sheet["B1"] = "dG_OH"
        sheet["C1"] = "z_CO_CO"
        sheet["D1"] = "z_CO_OH"
        sheet["E1"] = "z_OH_CO"
        sheet["F1"] = "z_OH_OH"
        sheet["G1"] = "Likelihood"
        sheet["A2",dim=1] = local_mins[:,1]
        sheet["B2",dim=1] = local_mins[:,2]
        sheet["C2",dim=1] = local_mins[:,3]
        sheet["D2",dim=1] = local_mins[:,4]
        sheet["E2",dim=1] = local_mins[:,5]
        sheet["F2",dim=1] = local_mins[:,6]
        sheet["G2",dim=1] = local_mins[:,7]
    end
end

function runOverall()

    material = "Ag90Pd10"
    global_opt_iter = 150
    local_opt_iter = 50

    # Unpack the priors and bounds here
    #################################
    
    # Pd single site LH lateral interaction priors and bounds
    #################################
    # 1. CO(g) + (*) <-> CO(*)
    dG0_CO_g_to_CO_ref_prior_mean = -0.85
    dG0_CO_g_to_CO_ref_bounds = (-1.2, 0.0)
    # 2. OH-(aq) + (*) <-> OH(*) + e-
    dG0_OH_aq_to_OH_ref_prior_mean = 0.93
    dG0_OH_aq_to_OH_ref_bounds = (0.0, 1.2)
    # CO-CO interaction param
    z_CO_CO_prior_mean = 0.5
    z_CO_CO_bounds = (0.0, 1.0)
    # CO-OH interaction param
    z_CO_OH_prior_mean = 0.2
    z_CO_OH_bounds = (0.0, 1.0)
    # OH-CO interaction param
    z_OH_CO_prior_mean = 0.1
    z_OH_CO_bounds = (0.0, 1.0)
    # OH-OH interaction param
    z_OH_OH_prior_mean = 0.1
    z_OH_OH_bounds = (0.0, 1.0)

    priors = [dG0_CO_g_to_CO_ref_prior_mean,  
              dG0_OH_aq_to_OH_ref_prior_mean,
              z_CO_CO_prior_mean,
              z_CO_OH_prior_mean,
              z_OH_CO_prior_mean,
              z_OH_OH_prior_mean]

    bounds = [dG0_CO_g_to_CO_ref_bounds, 
              dG0_OH_aq_to_OH_ref_bounds,
              z_CO_CO_bounds,
              z_CO_OH_bounds,
              z_OH_CO_bounds,
              z_OH_OH_bounds]

    ####################################################################
    
    controlFitting(global_opt_iter, local_opt_iter, priors, bounds)
end

runOverall()