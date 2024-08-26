using StatsBase, Random, Distributions, DataFrames, StaticArrays, ForwardDiff, DiffEqGPU, DifferentialEquations, CUDA


####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Function
###############################

function offspring!(offspring::Individual, parent::Individual)
    setfield!(offspring, :action, getfield(parent, :action))
    setfield!(offspring, :a, getfield(parent, :a))
    setfield!(offspring, :p, getfield(parent, :p))
    setfield!(offspring, :T, getfield(parent, :T))
    setfield!(offspring, :payoff, 0.0)
    setfield!(offspring, :interactions, 0)
end

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

function population_construction(parameters::SimulationParameters)
    individuals_dict = Dict{Int64, Individual}()
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
            indiv = Individual(action0 + rand(action0_dist), a0 + rand(a0_dist), p0 + rand(p0_dist), T0 + rand(T0_dist), 0.0, 0)
        else
            indiv = Individual(action0, a0, p0, T0, 0.0, 0)
        end

        individuals_dict[i] = indiv
    end

    return Population(parameters, individuals_dict, 0, 0)
end

function output!(outputs::DataFrame, t::Int64, pop::Population)
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

function benefit(actioni::Real, actionj, synergy::Real)
    sum_sqrt_actions = sqrt(max(actioni, 0.0)) + sum(sqrt.(max.(actionj, 0.0)))  # Handles both scalars and vectors
    sqrt_sum_actions = sqrt(max(actioni + sum(actionj), 0.0))  # Square root of the sum of all actions
    return (1 - synergy) * sum_sqrt_actions + synergy * sqrt_sum_actions
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

function payoff(actioni::Real, actionj, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(actioni, actionj, synergy) - cost(actioni) - external_punishment(actioni, norm_pool, punishment_pool)
end

function objective(actioni::Real, actionj, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return payoff(actioni, actionj, norm_pool, punishment_pool, synergy) - internal_punishment(actioni, norm_pool, T)
end

function objective_derivative(actioni::Real, actionj, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return ForwardDiff.derivative(x -> objective(x, actionj, norm_pool, punishment_pool, T, synergy), actioni)
end

function total_payoff!(ind1::Individual, ind2::Individual, norm_pool::Float64, punishment_pool::Float64, synergy::Float64)
    payoff1 = payoff(ind1.action, ind2.action, norm_pool, punishment_pool, synergy)
    payoff2 = payoff(ind2.action, ind1.action, norm_pool, punishment_pool, synergy)

    ind1.payoff = (payoff1 + ind1.interactions * ind1.payoff) / (ind1.interactions + 1)
    ind2.payoff = (payoff2 + ind2.interactions * ind2.payoff) / (ind2.interactions + 1)

    ind1.interactions += 1
    ind2.interactions += 1

    nothing
end

function total_payoff!(ind::Individual, synergy::Float64)
    payoff_ind = payoff(ind.action, ind.action, ind.a, ind.p, synergy)

    ind.payoff = (payoff_ind + ind.interactions * ind.payoff) / (ind.interactions + 1)

    ind.interactions += 1

    nothing
end

function fitness(ind::Individual)
    return ind.payoff - ind.p
end

function fitness(ind::Individual, fitness_scaling_factor_a::Float64, fitness_scaling_factor_b::Float64)
    return fitness_scaling_factor_a * exp(fitness(ind) * fitness_scaling_factor_b)
end


##################
# Behavioral Equilibrium Function
##################
#=
function remove_element(actions::SVector{N, T}, idx::Int) where {N, T}
    return SVector{N-1}(actions[1:idx-1]..., actions[idx+1:end]...)
end

function behav_ODE_static(u, p, t)
    # Construct the SVector using a tuple comprehension
    return SVector{length(u)}((objective_derivative(u[i], remove_element(u, i), p[1], p[2], p[3+i], p[3]) for i in 1:length(u))...)
end
=#

function behav_ODE_static(u, p, t)
    dx = objective_derivative(u[1], SVector{1}(u[2]), p[1], p[2], p[4], p[3])
    dy = objective_derivative(u[2], SVector{1}(u[1]), p[1], p[2], p[5], p[3])

    return SVector{2}(dx, dy)
end

function behav_eq(u0s, ps, tmax::Float64, num_groups::Int64)
    tspan = (0.0, tmax)

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each pair
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = num_groups, save_on = false)

    # Extract final action values
    final_actions = [sol[end] for sol in sim]

    return final_actions
end

function behav_eq!(groups::Vector{Vector{Individual}}, norm_pool::Float64, punishment_pool::Float64, synergy::Float64, tmax::Float64)
    tspan = (0.0, tmax)

    group_size = length(groups[1])
    num_groups = length(groups)

    u0s = Vector{SVector{group_size, Float32}}(undef, num_groups)
    ps = Vector{SVector{3 + group_size, Float32}}(undef, num_groups)

    # Extract initial conditions and parameters
    for (i, group) in enumerate(groups)
        # Collect initial actions
        actions = [Float32(group[j].action) for j in eachindex(group)]
        u0s[i] = SVector{group_size, Float32}(actions...)

        # Collect parameters
        T_values = [Float32(group[j].T) for j in eachindex(group)]
        ps[i] = SVector{3 + group_size, Float32}(norm_pool, punishment_pool, synergy, T_values...)
    end

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each group
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = num_groups, save_on = false)

    # Update action values
    final_actions = [sol[end] for sol in sim]
    for (group, action) in zip(groups, final_actions)
        setproperty!.(group, :action, action)
    end

    nothing
end


##################
# Social Interactions Function
##################

function update_norm_punishment_pools!(pop::Population)
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

function probabilistic_round(x::Float64)::Int
    lower = floor(Int, x)
    upper = ceil(Int, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

function shuffle_and_group(individuals_key::Vector{Int64}, population_size::Int64, group_size::Int64, relatedness::Float64)
    shuffle!(individuals_key)
    groups = Vector{Vector{Int64}}()

    # Iterate over each individual and form a group
    for i in 1:population_size
        focal_individual = individuals_key[i]

        # Create a list of potential candidates excluding the focal individual
        candidates = filter(x -> x != focal_individual, individuals_key)

        # Calculate the number of related individuals using probabilistic rounding
        num_related = probabilistic_round(relatedness * group_size)
        num_random = group_size - num_related

        if num_related > 0
            # Sample random individuals from the filtered candidates
            random_individuals = sample(candidates, num_random, replace=false)

            # Fill the group with related individuals and sampled individuals
            related_individuals = fill(focal_individual, num_related)
            final_group = [related_individuals; random_individuals]
        else
            # If relatedness is 0, just sample a full group
            random_individuals = sample(candidates, group_size - 1, replace=false)
            final_group = [focal_individual; random_individuals]
        end

        push!(groups, final_group)
    end

    num_groups = length(groups)
    return groups, num_groups
end

function collect_initial_conditions_and_parameters(groups::Vector{Vector{Int64}}, num_groups::Int64, group_size::Int64, pop::Population)
    u0s = Vector{SVector{group_size, Float32}}(undef, num_groups)
    ps = Vector{SVector{3 + group_size, Float32}}(undef, num_groups)

    for (i, group) in enumerate(groups)
        # Collect initial actions
        actions = [Float32(pop.individuals[idx].action) for idx in group]
        u0s[i] = SVector{group_size, Float32}(actions...)

        # Collect parameters
        T_values = [Float32(pop.individuals[idx].T) for idx in group]
        ps[i] = SVector{3 + group_size, Float32}(pop.norm_pool, pop.punishment_pool, pop.parameters.synergy, T_values...)
    end

    return u0s, ps
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{2, Float32}}, pairs::Vector{Tuple{Int64, Int64}}, pop::Population)
    for (i, (idx1, idx2)) in enumerate(pairs)
        ind1 = pop.individuals[idx1]
        ind2 = pop.individuals[idx2]
        ind1.action, ind2.action = final_actions[i]

        if idx1 == idx2
            total_payoff!(ind1, pop.parameters.synergy)
        else
            total_payoff!(ind1, ind2, pop.norm_pool, pop.punishment_pool, pop.parameters.synergy)
        end

        # Uncomment below to make the payoffs fixed
        # ind1.payoff = 1.0
        # ind2.payoff = 1.0
    end

    nothing
end

function social_interactions!(pop::Population)
    individuals_key = collect(keys(pop.individuals))

    # Update norm and punishment pools
    update_norm_punishment_pools!(pop)

    # Shuffle and pair individuals
    pairs, num_pairs = shuffle_and_pair(individuals_key, pop.parameters.population_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    #u0s, ps = collect_initial_conditions_and_parameters(pairs, num_pairs, pop)
    #final_actions = behav_eq(u0s, ps, pop.parameters.tmax, num_pairs)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, pairs, pop)

    nothing
end


##################
# Reproduction Function
##################

function reproduce!(pop::Population)
    # Calculate fitness
    fitnesses = map(individual -> fitness(individual, pop.parameters.fitness_scaling_factor_a, pop.parameters.fitness_scaling_factor_b), values(pop.individuals))
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
function reproduce!(pop::Population)
    # Calculate fitness
    fitnesses = map(individual -> fitness(individual, pop.parameters.fitness_scaling_factor_a, pop.parameters.fitness_scaling_factor_b), values(pop.individuals))
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

function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
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
function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
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

function simulation(pop::Population)

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

        # Per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(outputs, t, pop)
        end

        # Reproduction function to produce new generation
        reproduce!(pop)

        # Mutation function iterates over population and mutates at chance probability μ
        if pop.parameters.mutation_rate > 0
            mutate!(pop, truncate_bounds)
        end
    end

    return outputs
end