using StatsBase, Random, Distributions, DataFrames, StaticArrays, ForwardDiff, DiffEqGPU, DifferentialEquations, CUDA


####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Funcs #
###############################

    # Create an initial population
    # Format output

function truncation_bounds(variance::Float64, retain_proportion::Float64)
    # Calculate tail probability alpha
    alpha = 1 - retain_proportion

    # Calculate z-score corresponding to alpha/2
    z_alpha_over_2 = quantile(Normal(), 1 - alpha/2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * sqrt(variance)
    upper_bound = z_alpha_over_2 * sqrt(variance)

    return SA[lower_bound, upper_bound]
end

function population_construction(parameters::simulation_parameters)
    individuals_dict = Dict{Int64, individual}()
    trait_var = parameters.trait_var
    use_distribution = trait_var != 0

    # Collect initial traits
    action0 = parameters.action0
    a0 = parameters.a0
    p0 = parameters.p0
    T0 = parameters.T0

    # Construct distributions if necessary
    if use_distribution
        lower_bound, upper_bound = truncation_bounds(trait_var, 0.99)
        action0_dist = truncated(Normal(0, trait_var), lower=max(lower_bound, -action0), upper=upper_bound)
        a0_dist = truncated(Normal(0, trait_var), lower=max(lower_bound, -a0), upper=upper_bound)
        p0_dist = truncated(Normal(0, trait_var), lower=max(lower_bound, -p0), upper=upper_bound)
        T0_dist = truncated(Normal(0, trait_var), lower=max(lower_bound, -T0), upper=upper_bound)
    end

    # Create individuals
    for i in 1:parameters.N
        if use_distribution
            action0 += rand(action0_dist)
            a0 += rand(a0_dist)
            p0 += rand(p0_dist)
            T0 += rand(T0_dist)
        end
        indiv = individual(action0, a0, p0, T0, 0.0, 0)
        individuals_dict[i] = indiv
    end

    return population(parameters, individuals_dict, 0, 0)
end

function output!(outputs::DataFrame, t::Int64, pop::population)
    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * pop.parameters.N + 1
    end

    # Preallocate vectors for batch assignment
    N = pop.parameters.N
    generation_col = fill(t, N)
    individual_col = 1:N
    action_col = Vector{Float64}(undef, N)
    a_col = Vector{Float64}(undef, N)
    p_col = Vector{Float64}(undef, N)
    T_col = Vector{Float64}(undef, N)
    payoff_col = Vector{Float64}(undef, N)

    # Collect the data for each individual
    for i in 1:N
        ind = pop.individuals[i]
        action_col[i] = ind.action
        a_col[i] = ind.a
        p_col[i] = ind.p
        T_col[i] = ind.T
        payoff_col[i] = ind.payoff
    end

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base + N - 1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = generation_col
    outputs.individual[output_rows] = individual_col
    outputs.action[output_rows] = action_col
    outputs.a[output_rows] = a_col
    outputs.p[output_rows] = p_col
    outputs.T[output_rows] = T_col
    outputs.payoff[output_rows] = payoff_col

    nothing
end


##################
# Fitness function
##################

    # Calculate payoff, and keep a running average of payoff for each individual
    # After each session of interaction the running average becomes the individual's payoff

function benefit(action1::Real, action2::Float64, v::Float64)
    sqrt_action1 = √max(action1, 0)
    sqrt_action2 = √max(action2, 0)
    sqrt_sum = √max((action1 + action2), 0)
    return (1 - v) * (sqrt_action1 + sqrt_action2) + v * sqrt_sum
end

function cost(action::Real)
    return action^2
end

function external_punishment(action::Real, norm_pool::Float64, punishment_pool::Float64)
    return punishment_pool * (action - norm_pool)^2
end

function internal_punishment(action::Real, norm_pool::Float64, T::Float64)
    return T * (action - norm_pool)^2
end

function payoff(action1::Real, action2::Float64, norm_pool::Float64, punishment_pool::Float64, v::Float64)
    return benefit(action1, action2, v) - cost(action1) - external_punishment(action1, norm_pool, punishment_pool)
end

function objective(action1::Real, action2::Float64, norm_pool::Float64, punishment_pool::Float64, T::Float64, v::Float64)
    return payoff(action1, action2, norm_pool, punishment_pool, v) - internal_punishment(action1, norm_pool, T)
end

function objective_derivative(action1::Real, action2::Float64, norm_pool::Float64, punishment_pool::Float64, T::Float64, v::Float64)
    return ForwardDiff.derivative(x -> objective(x, action2, norm_pool, punishment_pool, T, v), action1)
end

function total_payoff!(ind1::individual, ind2::individual, norm_pool::Float64, punishment_pool::Float64, v::Float64)
    payoff1 = max(payoff(ind1.action, ind2.action, norm_pool, punishment_pool, v), 0)
    payoff2 = max(payoff(ind2.action, ind1.action, norm_pool, punishment_pool, v), 0)

    ind1.payoff = (payoff1 + ind1.interactions * ind1.payoff) / (ind1.interactions + 1)
    ind2.payoff = (payoff2 + ind2.interactions * ind2.payoff) / (ind2.interactions + 1)

    ind1.interactions += 1
    ind2.interactions += 1

    nothing
end


##################
# Behavioral Equilibrium function
##################

function behav_ODE_static(u, p, t)
    dx = objective_derivative(u[1], u[2], p[1], p[2], p[3], p[5])
    dy = objective_derivative(u[2], u[1], p[1], p[2], p[4], p[5])

    return SA[dx, dy]
end

function behav_eq(u0s::Array{SArray{Tuple{2}, Float64}}, ps::Array{SArray{Tuple{5}, Float64}}, tmax::Float64)
    tspan = (0.0, tmax)

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each pair
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = length(u0s), save_on = false)

    # Extract final action values
    final_actions = [sol[end] for sol in sim]

    return final_actions
end

function behav_eq!(pairs::Vector{Tuple{individual, individual}}, norm_pool::Float64, punishment_pool::Float64, tmax::Float64, v::Float64)
    # Extract initial conditions and parameters
    u0s = [SA[ind1.action, ind2.action] for (ind1, ind2) in pairs]
    tspan = (0.0, tmax)
    ps = [SA[norm_pool, punishment_pool, ind1.T, ind2.T, v] for (ind1, ind2) in pairs]

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each pair
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = length(pairs), save_on = false)

    # Update action values
    final_actions = [sol[end] for sol in sim]
    for ((ind1, ind2), action) in zip(pairs, final_actions)
        ind1.action, ind2.action = action
    end

    nothing
end


##################
# Social Interactions function
##################

    # Pair individuals with the possibiliy of pairing more than once
    # Everyone has the same chance of picking a partner / getting picked
    # At the end of the day everyone is picked roughly an equal number of times

function social_interactions!(pop::population)
    individuals_key = collect(keys(pop.individuals))
    individuals_shuffle = shuffle(individuals_key)

    # If the number of individuals is odd, append a random individual to the shuffled list
    if isodd(length(individuals_shuffle))
        push!(individuals_shuffle, individuals_key[rand(1:length(individuals_key))])
    end

    num_pairs = length(individuals_shuffle) ÷ 2

    # Prepare storage for initial conditions and parameters for all pairs
    u0s = Vector{SArray{Tuple{2}, Float64}}(undef, num_pairs)
    ps = Vector{SArray{Tuple{5}, Float64}}(undef, num_pairs)

    # Initialize sums for calculating mean
    norm_sum = 0.0
    punishment_sum = 0.0

    # Iterate through individuals to sum norms and punishments
    for individual in values(pop.individuals)
        norm_sum += individual.a
        punishment_sum += individual.p
    end

    # Update norm and punishment pools
    pop.norm_pool = norm_sum / pop.parameters.N
    pop.punishment_pool = punishment_sum / pop.parameters.N

    # Collect initial conditions and parameters for all pairs
    for i in 1:num_pairs
        ind1 = pop.individuals[individuals_shuffle[2i-1]]
        ind2 = pop.individuals[individuals_shuffle[2i]]
        u0s[i] = SA[ind1.action, ind2.action]
        ps[i] = SA[pop.norm_pool, pop.punishment_pool, ind1.T, ind2.T, pop.parameters.v]
    end

    # Calculate final actions for all pairs
    final_actions = behav_eq(u0s, ps, pop.parameters.tmax)

    # Update actions and payoffs for all pairs based on final actions
    for i in 1:num_pairs
        ind1 = pop.individuals[individuals_shuffle[2i-1]]
        ind2 = pop.individuals[individuals_shuffle[2i]]
        ind1.action, ind2.action = copy(final_actions[i])
        total_payoff!(ind1, ind2, pop.norm_pool, pop.punishment_pool, pop.parameters.v)
    end

    nothing
end


##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents
    # number of individuals in population remains the same

function reproduce!(pop::population)
    payoffs = map(individual -> individual.payoff, values(pop.individuals))
    keys_list = collect(keys(pop.individuals))

    # Sample with the given weights
    sampled_keys = sample(keys_list, ProbabilityWeights(payoffs), pop.parameters.N, replace=true, ordered=false)

    # Sort keys
    sort!(sampled_keys)

    # Update population individuals based on sampled keys
    for (key, sampled_key) in zip(1:pop.parameters.N, sampled_keys)
        copy!(pop.individuals[key], pop.individuals[sampled_key])
    end

    nothing
end


##################
#  Mutation Function 
##################

    # offspring have slightly different trait values from their parents
    # use an independent draw function for each of the traits that could mutate

function mutate!(pop::population, truncate_bounds::SArray{Tuple{2}, Float64})
    mut_var = pop.parameters.mut_var

    # Only mutate if necessary
    if mut_var == 0
        return nothing
    end

    u = pop.parameters.u
    #lower_bound, upper_bound = truncate_bounds

    lower_bound, upper_bound = truncate_bounds

    # Indpendent draw for each of the traits to mutate
    for ind in values(pop.individuals)
        if rand() <= u
            a_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -ind.a), upper=upper_bound)
            ind.a += rand(a_dist)
        end

        if rand() <= u
            p_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -ind.p), upper=upper_bound)
            ind.p += rand(p_dist)
        end

        if rand() <= u
            T_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -ind.T), upper=upper_bound)
            ind.T += rand(T_dist)
        end
    end

    nothing
end


#######################
# Simulation Function #
#######################

function simulation(pop::population)

    ############
    # Sim init #
    ############

    output_length = floor(Int64, pop.parameters.gmax/pop.parameters.output_save_tick) * pop.parameters.N
    outputs = DataFrame(
        generation = Vector{Int64}(undef, output_length),
        individual = Vector{Int64}(undef, output_length),
        action = Vector{Float64}(undef, output_length),
        a = Vector{Float64}(undef, output_length),
        p = Vector{Float64}(undef, output_length),
        T = Vector{Float64}(undef, output_length),
        payoff = Vector{Float64}(undef, output_length)
    )

    truncate_bounds = truncation_bounds(pop.parameters.mut_var, 0.99)

    ############
    # Sim Loop #
    ############

    for t in 1:pop.parameters.gmax
        # Execute social interactions and calculate payoffs
        social_interactions!(pop)

        # Reproduction function to produce new generation
        reproduce!(pop)

        # Mutation function iterates over population and mutates at chance probability μ
        if pop.parameters.u > 0
            mutate!(pop, truncate_bounds)
        end

        # Per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(outputs, t, pop)
        end
    end

    return outputs
end