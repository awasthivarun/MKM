using StatsModels
using DataFrames
using GLM
using Plots
using NLsolve
Eref=0.260 # Reference potential for the CO oxidation reaction in V vs. RHE
kB = 8.617333262145e-5 # Boltzmann constant in eV/K
T = 298.15 # Temperature in K
h = 4.135667696e-15 # Planck's constant in eV

function simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, E_array, P_CO_array, a_OH_array, orderType)

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
                        driving_CO = (K_CO_0 * P_CO_array[j])
                        driving_OH = (K_OH_0 * a_OH_array[i])

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
                        driving_CO = (K_CO_0 * P_CO_array[j])
                        driving_OH = (K_OH_0 * a_OH_array[i])

                        # All denominators corresponding to each site type are the same.
                        denom = 1 + driving_CO + driving_OH

                        jvec[1,1] = denom
                        jvec[1,2] = 0
                        jvec[2,1] = 0
                        jvec[2,2] = denom
                    
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

                ddG_LH[j,:] .= 0

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
                if a_OH_array[j] == 0.19
                    E_array[:] = E_array[:] .- 0.059 * (log10(0.5) - log10(0.25))
                elseif a_OH_array[j] == 0.76
                    E_array[:] = E_array[:] .+ 0.059 * (log10(1) - log10(0.5))
                else
                    E_array[:] = E_array
                end
                for k = 1:length(E_array)
                    # Update the initial conditions using the solved coverages at the previous potential
                    if k == 1
                        IC = initialConditionFirst
                    else
                        IC = [t_CO_array_for_OH_order[j,k-1], t_OH_array_for_OH_order[j,k-1]] 
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
                        driving_CO = (K_CO_0 * P_CO_array[i])
                        driving_OH = (K_OH_0 * a_OH_array[j])

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
                        driving_CO = (K_CO_0 * P_CO_array[i])
                        driving_OH = (K_OH_0 * a_OH_array[j])

                        # All denominators corresponding to each site type are the same.
                        denom = 1 + driving_CO + driving_OH

                        jvec[1,1] = denom
                        jvec[1,2] = 0
                        jvec[2,1] = 0
                        jvec[2,2] = denom
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
                ddG_LH[j,:] .= 0

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


# Example input arrays (replace with your actual data)
E_array = range(0, stop=0.5, length=50)
P_CO_array = 1e5*[0.001, 0.01, 0.1, 1]
a_OH_array = [0.1, 0.3, 1.0]*1e3
dG0_CO_g_to_CO_ref = -0.5
dG0_OH_aq_to_OH_ref = 0.0
orderType = "CO order"

# Call your function
alpha_250, alpha_500, alpha_1000, CO_order_250, CO_order_500, CO_order_1000, _, _, _ = 
    simulateMacrokineticObservables(dG0_CO_g_to_CO_ref, dG0_OH_aq_to_OH_ref, E_array, P_CO_array, a_OH_array, orderType)

plt1 = plot(E_array, CO_order_250[1, :], label="0.1-1%")
plot!(plt1, E_array, CO_order_250[2, :], label="1-10")
plot!(plt1, E_array, CO_order_250[3, :], label="10-100")
xlabel!("Potential (V)")
ylabel!("CO Order")
title!("CO Order vs. Potential")
display(plt1)  # Show first plot

plt2 = plot(E_array, OH_order[1, :], label="0.1-1%")
xlabel!("Potential (V)")
ylabel!("OH Order")
title!("OH Order vs. Potential")
display(plt2)  # Show second plot