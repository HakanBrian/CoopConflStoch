using StatsBase, Random, Distributions, DataFrames, StaticArrays, ForwardDiff, DiffEqGPU, DifferentialEquations, CUDA


####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Function
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

function offspring!(offspring::individual, parent::individual)
    setfield!(offspring, :action, getfield(parent, :action))
    setfield!(offspring, :a, getfield(parent, :a))
    setfield!(offspring, :p, getfield(parent, :p))
    setfield!(offspring, :T, getfield(parent, :T))
    setfield!(offspring, :payoff, 0.0)
    setfield!(offspring, :interactions, 0)
end

function population_construction(parameters::simulation_parameters)
    individuals_dict = Dict{Int64, individual}()
    trait_variance = parameters.trait_variance
    use_distribution = trait_variance != 0

    # Collect initial traits
    action0 = parameters.action0
    a0 = parameters.a0
    p0 = parameters.p0
    T0 = parameters.T0

    # Construct distributions if necessary
    if use_distribution
        lower_bound, upper_bound = truncation_bounds(trait_variance, 0.99)
        action0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -action0), upper=upper_bound)
        a0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -a0), upper=upper_bound)
        p0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -p0), upper=upper_bound)
        T0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -T0), upper=upper_bound)
    end

    # Create individuals
    for i in 1:parameters.population_size
        if use_distribution
            indiv = individual(action0 + rand(action0_dist), a0 + rand(a0_dist), p0 + rand(p0_dist), T0 + rand(T0_dist), 0.0, 0)
        else
            indiv = individual(action0, a0, p0, T0, 0.0, 0)
        end

        individuals_dict[i] = indiv
    end

    return population(parameters, individuals_dict, 0, 0)
end

function output!(outputs::DataFrame, t::Int64, pop::population)
    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * pop.parameters.population_size + 1
    end

    # Preallocate vectors for batch assignment
    N = pop.parameters.population_size
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
# Fitness Function
##################

    # Calculate payoff, and keep a running average of payoff for each individual
    # After each session of interaction the running average becomes the individual's payoff

function benefit(action1::Real, action2::Real, synergy::Real)
    sqrt_action1 = √action1
    sqrt_action2 = √action2
    sqrt_sum = √(action1 + action2)
    return (1 - synergy) * (sqrt_action1 + sqrt_action2) + synergy * sqrt_sum
end

function cost(action::Real)
    return action^2
end

function external_punishment(action::Real, norm_pool::Real, punishment_pool::Real)
    return punishment_pool * (action - norm_pool)^2
end

function internal_punishment(action::Real, norm_pool::Real, T::Real)
    return T * (action - norm_pool)^2
end

function payoff(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(action1, action2, synergy) - cost(action1) - external_punishment(action1, norm_pool, punishment_pool)
end

function objective(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return payoff(action1, action2, norm_pool, punishment_pool, synergy) - internal_punishment(action1, norm_pool, T)
end

function objective_derivative(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return ForwardDiff.derivative(x -> objective(x, action2, norm_pool, punishment_pool, T, synergy), action1)
end

function total_payoff!(ind1::individual, ind2::individual, norm_pool::Float64, punishment_pool::Float64, synergy::Float64)
    payoff1 = payoff(ind1.action, ind2.action, norm_pool, punishment_pool, synergy)
    payoff2 = payoff(ind2.action, ind1.action, norm_pool, punishment_pool, synergy)

    ind1.payoff = (payoff1 + ind1.interactions * ind1.payoff) / (ind1.interactions + 1)
    ind2.payoff = (payoff2 + ind2.interactions * ind2.payoff) / (ind2.interactions + 1)

    ind1.interactions += 1
    ind2.interactions += 1

    nothing
end

function total_payoff!(ind::individual, synergy::Float64)
    payoff_ind = payoff(ind.action, ind.action, ind.a, ind.p, synergy)

    ind.payoff = (payoff_ind + ind.interactions * ind.payoff) / (ind.interactions + 1)

    ind.interactions += 1

    nothing
end

function fitness(ind::individual)
    return ind.payoff - ind.p
end

function fitness(ind::individual, fitness_scaling_factor::Float64)
    return 0.004 * exp(fitness(ind) * fitness_scaling_factor)
end


##################
# Behavioral Equilibrium Function
##################

function behav_ODE_static(u, p, t)
    dx = objective_derivative(u[1], u[2], p[1], p[2], p[3], p[5])
    dy = objective_derivative(u[2], u[1], p[1], p[2], p[4], p[5])

    return SA[dx, dy]
end

function behav_eq(u0s::Array{SArray{Tuple{2}, Float32}}, ps::Array{SArray{Tuple{5}, Float32}}, tmax::Float32, num_pairs::Int64)
    tspan = (0.0f0, tmax)

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each pair
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = num_pairs, save_on = false)

    # Extract final action values
    final_actions = [sol[end] for sol in sim]

    return final_actions
end

function behav_eq!(pairs::Vector{Tuple{individual, individual}}, norm_pool::Float64, punishment_pool::Float64, tmax::Float32, synergy::Float64)
    # Extract initial conditions and parameters
    u0s = [SA_F32[ind1.action, ind2.action] for (ind1, ind2) in pairs]
    tspan = (0.0f0, tmax)
    ps = [SA_F32[norm_pool, punishment_pool, ind1.T, ind2.T, synergy] for (ind1, ind2) in pairs]

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
# Social Interactions Function
##################

    # Pair individuals with the possibiliy of pairing more than once
    # Everyone has the same chance of picking a partner / getting picked
    # At the end of the day everyone is picked roughly an equal number of times

function update_norm_punishment_pools!(pop::population)
    norm_sum = 0.0
    punishment_sum = 0.0
    for individual in values(pop.individuals)
        norm_sum += individual.a
        punishment_sum += individual.p
    end

    # Update norm and punishment pools
    pop.norm_pool = norm_sum / pop.parameters.population_size
    pop.punishment_pool = punishment_sum / pop.parameters.population_size

    nothing
end

function shuffle_and_pair(individuals_key::Vector{Int64}, population_size::Int64, relatedness::Float64)
    shuffle!(individuals_key)
    pairs = Vector{Tuple{Int64, Int64}}()
    i = 1

    while i <= population_size
        if i == population_size
            # Handle the last individual separately
            if rand() <= relatedness
                push!(pairs, (individuals_key[i], individuals_key[i]))
            else
                push!(pairs, (individuals_key[i], rand(individuals_key[1:i-1])))
            end
            i += 1
        elseif rand() <= relatedness
            push!(pairs, (individuals_key[i], individuals_key[i]))
            i += 1
        else
            push!(pairs, (individuals_key[i], individuals_key[i+1]))
            i += 2
        end
    end

    num_pairs = length(pairs)

    return pairs, num_pairs
end

function collect_initial_conditions_and_parameters(pairs::Vector{Tuple{Int64, Int64}}, num_pairs::Int64, pop::population)
    u0s = Vector{SArray{Tuple{2}, Float32}}(undef, num_pairs)
    ps = Vector{SArray{Tuple{5}, Float32}}(undef, num_pairs)

    for (i, (idx1, idx2)) in enumerate(pairs)
        ind1 = pop.individuals[idx1]
        ind2 = pop.individuals[idx2]
        u0s[i] = SA_F32[ind1.action; ind2.action]

        if idx1 == idx2
            ps[i] = SA_F32[ind1.a, ind1.p, ind1.T, ind1.T, pop.parameters.synergy]
        else
            ps[i] = SA_F32[pop.norm_pool, pop.punishment_pool, ind1.T, ind2.T, pop.parameters.synergy]
        end
    end

    return u0s, ps
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{2, Float32}}, pairs::Vector{Tuple{Int64, Int64}}, pop::population)
    for (i, (idx1, idx2)) in enumerate(pairs)
        ind1 = pop.individuals[idx1]
        ind2 = pop.individuals[idx2]
        ind1.action, ind2.action = final_actions[i]

        if idx1 == idx2
            total_payoff!(ind1, pop.parameters.synergy)
        else
            total_payoff!(ind1, ind2, pop.norm_pool, pop.punishment_pool, pop.parameters.synergy)
        end

        # Uncomment below to fix payoffs
        # ind1.payoff = 1.0
        # ind2.payoff = 1.0
    end

    nothing
end

function social_interactions!(pop::population)
    individuals_key = collect(keys(pop.individuals))

    # Update norm and punishment pools
    update_norm_punishment_pools!(pop)

    # Shuffle and pair individuals
    pairs, num_pairs = shuffle_and_pair(individuals_key, pop.parameters.population_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    u0s, ps = collect_initial_conditions_and_parameters(pairs, num_pairs, pop)
    final_actions = behav_eq(u0s, ps, pop.parameters.tmax, num_pairs)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, pairs, pop)

    nothing
end


##################
# Reproduction Function
##################

    # Offspring inherit the payoff and traits of the parent
    # Number of individuals in population remains the same

function reproduce!(pop::population)
    # Calculate fitness
    fitness_scaling_factor = pop.parameters.fitness_scaling_factor
    fitnesses = map(individual -> fitness(individual, fitness_scaling_factor), values(pop.individuals))
    keys_list = collect(keys(pop.individuals))

    # Sample with the given weights
    sampled_keys = sample(keys_list, ProbabilityWeights(fitnesses), pop.parameters.population_size, replace=true, ordered=false)

    # Sort keys
    sort!(sampled_keys)

    # Update population individuals based on sampled keys
    for (key, sampled_key) in enumerate(sampled_keys)
        offspring!(pop.individuals[key], pop.individuals[sampled_key])
    end

    nothing
end

#= Maximal fitness reproduction
function reproduce!(pop::population)
    # Calculate fitness
    fitness_scaling_factor = pop.parameters.fitness_scaling_factor
    fitnesses = map(individual -> fitness(individual, fitness_scaling_factor), values(pop.individuals))
    keys_list = collect(keys(pop.individuals))

    # Find the highest fitness individual
    highest_fitness_individual = pop.individuals[keys_list[argmax(fitnesses)]]

    # Update population individuals based on maximal fitness
    for i in 1:pop.parameters.population_size
        copy!(pop.individuals[i], highest_fitness_individual)
    end

    nothing
end
=#


##################
# Mutation Function 
##################

    # Offspring have slightly different trait values from their parents
    # Use an independent draw function for each of the traits that could mutate

function mutate!(pop::population, truncate_bounds::SArray{Tuple{2}, Float64})
    mutation_variance = pop.parameters.mutation_variance

    # Only mutate if necessary
    if mutation_variance == 0
        return nothing
    end

    mutation_rate = pop.parameters.mutation_rate
    lower_bound, upper_bound = truncate_bounds

    # Indpendent draw for each of the traits to mutate
    for ind in values(pop.individuals)
        if rand() <= mutation_rate
            a_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -ind.a), upper=upper_bound)
            ind.a += rand(a_dist)
        end

        if rand() <= mutation_rate
            p_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -ind.p), upper=upper_bound)
            ind.p += rand(p_dist)
        end

        # if rand() <= mutation_rate
        #     T_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -ind.T), upper=upper_bound)
        #     ind.T += rand(T_dist)
        # end
    end

    nothing
end

#= Mutation units
function mutate!(pop::population, truncate_bounds::SArray{Tuple{2}, Float64})
    mutation_unit = pop.parameters.mutation_variance

    # Only mutate if necessary
    if mutation_unit == 0
        return nothing
    end

    mutation_direction = [-1, 1]
    mutation_rate = pop.parameters.mutation_rate

    # Indpendent draw for each of the traits to mutate
    for ind in values(pop.individuals)
        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit

            if ind.a + mutation_amount <= 0
                ind.a = 0
            else
                ind.a += mutation_amount
            end
        end

        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit

            if ind.p + mutation_amount <= 0
                ind.p = 0
            else
                ind.p += mutation_amount
            end
        end

        # Uncomment below if 'T' trait mutation is required
        # if rand() <= mutation_rate
        #   mutation_amount = rand(mutation_direction) * mutation_unit
        #
        #    if ind.T + mutation_amount <= 0
        #        ind.T = 0
        #    else
        #        ind.T += mutation_amount
        #    end
        # end
    end

    nothing
end
=#


#######################
# Simulation Function #
#######################

function simulation(pop::population)

    ############
    # Sim init #
    ############

    output_length = floor(Int64, pop.parameters.gmax/pop.parameters.output_save_tick) * pop.parameters.population_size
    outputs = DataFrame(
        generation = Vector{Int64}(undef, output_length),
        individual = Vector{Int64}(undef, output_length),
        action = Vector{Float64}(undef, output_length),
        a = Vector{Float64}(undef, output_length),
        p = Vector{Float64}(undef, output_length),
        T = Vector{Float64}(undef, output_length),
        payoff = Vector{Float64}(undef, output_length)
    )

    truncate_bounds = truncation_bounds(pop.parameters.mutation_variance, 0.99)

    ############
    # Sim Loop #
    ############

    for t in 1:pop.parameters.gmax
        # Execute social interactions and calculate payoffs
        social_interactions!(pop)

        # Reproduction function to produce new generation
        reproduce!(pop)

        # Mutation function iterates over population and mutates at chance probability μ
        if pop.parameters.mutation_rate > 0
            mutate!(pop, truncate_bounds)
        end

        # Per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(outputs, t, pop)
        end
    end

    return outputs
end