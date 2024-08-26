using StatsBase, Random, Distributions, DataFrames, StaticArrays, ForwardDiff, DiffEqGPU, DifferentialEquations, CUDA


####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Function
###############################

function offspring!(pop::Population, offspring_index::Int64, parent_index::Int64)
    # Copy traits from parent to offspring
    pop.action[offspring_index] = pop.action[parent_index]
    pop.norm[offspring_index] = pop.norm[parent_index]
    pop.ext_pun[offspring_index] = pop.ext_pun[parent_index]
    pop.int_pun[offspring_index] = pop.int_pun[parent_index]

    # Set initial values for offspring
    pop.payoff[offspring_index] = 0.0f0
    pop.interactions[offspring_index] = 0
end

function truncation_bounds(variance::Float64, retain_proportion::Float64)
    # Calculate tail probability alpha
    alpha = 1 - retain_proportion

    # Calculate z-score corresponding to alpha/2
    z_alpha_over_2 = quantile(Normal(), 1 - alpha/2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * √variance
    upper_bound = z_alpha_over_2 * √variance

    return SA[lower_bound, upper_bound]
end

function population_construction(parameters::SimulationParameters)
    trait_variance = parameters.trait_variance
    use_distribution = trait_variance != 0

    # Collect initial traits
    action0 = parameters.action0
    norm0 = parameters.norm0
    ext_pun0 = parameters.ext_pun0
    int_pun0 = parameters.int_pun0
    pop_size = parameters.population_size

    # Initialize arrays for attributes
    actions = Vector{Float32}(undef, pop_size)
    norms = Vector{Float32}(undef, pop_size)
    ext_puns = Vector{Float32}(undef, pop_size)
    int_puns = Vector{Float32}(undef, pop_size)
    payoffs = Vector{Float32}(undef, pop_size)
    interactions = Vector{Int64}(undef, pop_size)

    # Construct distributions if necessary
    if use_distribution
        lower_bound, upper_bound = truncation_bounds(trait_variance, 0.99)
        action0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -action0), upper=upper_bound)
        norm0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -norm0), upper=upper_bound)
        ext_pun0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -ext_pun0), upper=upper_bound)
        int_pun0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -int_pun0), upper=upper_bound)
    end

    # Create individuals
    for i in 1:pop_size
        if use_distribution
            actions[i] = action0 + rand(action0_dist)
            norms[i] = norm0 + rand(norm0_dist)
            ext_puns[i] = ext_pun0 + rand(ext_pun0_dist)
            int_puns[i] = int_pun0 + rand(int_pun0_dist)
        else
            actions[i] = action0
            norms[i] = norm0
            ext_puns[i] = ext_pun0
            int_puns[i] = int_pun0
        end
        payoffs[i] = 0.0f0
        interactions[i] = 0
    end

    return Population(
        parameters,
        actions,
        norms,
        ext_puns,
        int_puns,
        payoffs,
        interactions,
        0.0f0,
        0.0f0
    )
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
    action_col = pop.action
    norm_col = pop.norm
    ext_pun_col = pop.ext_pun
    int_pun_col = pop.int_pun
    payoff_col = pop.payoff

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base + N - 1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = generation_col
    outputs.individual[output_rows] = individual_col
    outputs.action[output_rows] = action_col
    outputs.a[output_rows] = norm_col
    outputs.p[output_rows] = ext_pun_col
    outputs.T[output_rows] = int_pun_col
    outputs.payoff[output_rows] = payoff_col

    nothing
end


##################
# Fitness Function
##################

function benefit(action_i::Real, actions_j, synergy::Real)
    sum_sqrt_actions = sqrt(max(action_i, 0.0)) + sum(sqrt.(max.(actions_j, 0.0)))
    sqrt_sum_actions = sqrt(max(action_i + sum(actions_j), 0.0))
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

function payoff(action_i::Real, actions_j, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(action_i, actions_j, synergy) - cost(action_i) - external_punishment(action_i, norm_pool, punishment_pool)
end

function objective(action_i::Real, actions_j, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return payoff(action_i, actions_j, norm_pool, punishment_pool, synergy) - internal_punishment(action_i, norm_pool, T)
end

function objective_derivative(action_i::Real, actions_j::SVector{N, <:Real}, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real) where N
    return ForwardDiff.derivative(x -> objective(x, actions_j, norm_pool, punishment_pool, T, synergy), action_i)
end

function total_payoff!(group_indices::Vector{Int64}, pop::Population)
    group_size = pop.parameters.group_size

    for i in 1:group_size
        # Skip processing if current individual is a virtual copy
        if i < group_size && group_indices[i] == group_indices[i + 1]
            continue
        end

        # Extract the action of the focal individual
        action_i = pop.action[group_indices[i]]

        # Collect actions from the other individuals in the group
        actions_j = [pop.action[group_indices[j]] for j in 1:group_size if j != i]

        # Compute the payoff for the focal individual
        payoff_foc = payoff(action_i, actions_j, pop.norm_pool, pop.pun_pool, pop.parameters.synergy)

        # Update the individual's payoff and interactions
        idx = group_indices[i]
        pop.payoff[idx] = (payoff_foc + pop.interactions[idx] * pop.payoff[idx]) / (pop.interactions[idx] + 1)
        pop.interactions[idx] += 1
    end

    nothing
end

function fitness(pop::Population, idx::Int)
    return pop.payoff[idx] - pop.ext_pun[idx]
end

function fitness(pop::Population, idx::Int, fitness_scaling_factor_a::Float64, fitness_scaling_factor_b::Float64)
    base_fitness = fitness(pop, idx)
    return fitness_scaling_factor_a * exp(base_fitness * fitness_scaling_factor_b)
end


##################
# Behavioral Equilibrium Function
##################

function remove_element(actions::SVector{N, T}, idx::Int64) where {N, T}
    return SVector{N-1}(ntuple(i -> i < idx ? actions[i] : actions[i + 1], N - 1))
end

function behav_ODE_static(u::SVector{N, T}, p::SVector, t) where {N, T}
    du = ntuple(i -> objective_derivative(u[i], remove_element(u, i), p[1], p[2], p[3 + i], p[3]), N)
    return SVector{N}(du)
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

function behav_eq!(groups::Vector{Vector{Int64}}, pop::Population, tmax::Float64)
    tspan = (0.0, tmax)
    
    group_size = length(groups[1])
    num_groups = length(groups)

    u0s = Vector{SVector{group_size, Float32}}(undef, num_groups)
    ps = Vector{SVector{3 + group_size, Float32}}(undef, num_groups)

    # Extract initial conditions and parameters
    for (i, group_indices) in enumerate(groups)
        # Collect initial actions using SoA approach
        actions = [Float32(pop.action[idx]) for idx in group_indices]
        u0s[i] = SVector{group_size, Float32}(actions...)

        # Collect parameters using SoA approach
        int_pun_values = [pop.int_pun[idx] for idx in group_indices]
        ps[i] = SVector{3 + group_size, Float32}(pop.norm_pool, pop.pun_pool, pop.parameters.synergy, int_pun_values...)
    end

    # Initialize a problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0s[1], tspan, ps[1])

    # Function to remake the problem for each group
    prob_func = (prob, i, repeat) -> remake(prob, u0 = u0s[i], p = ps[i])

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem
    sim = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = num_groups, save_on = false)

    # Update action values using the SoA approach
    final_actions = [sol[end] for sol in sim]
    for (group_indices, action) in zip(groups, final_actions)
        for (idx, action_value) in zip(group_indices, action)
            pop.action[idx] = action_value
        end
    end

    nothing
end


##################
# Social Interactions Function
##################

function update_norm_punishment_pools!(pop::Population)
    # Update norm and punishment pools
    pop.norm_pool = mean(pop.norm)
    pop.pun_pool = mean(pop.ext_pun)

    nothing
end

function probabilistic_round(x::Float64)::Int64
    lower = floor(Int64, x)
    upper = ceil(Int64, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

function shuffle_and_group(population_size::Int64, group_size::Int64, relatedness::Float64)
    individuals_key = collect(1:population_size)
    shuffle!(individuals_key)
    groups = Vector{Vector{Int64}}(undef, population_size)

    # Iterate over each individual index and form a group
    for i in 1:population_size
        focal_individual_index = individuals_key[i]

        # Create a list of potential candidates excluding the focal individual
        candidates = filter(x -> x != focal_individual_index, individuals_key)

        # Calculate the number of related individuals using probabilistic rounding
        num_related = probabilistic_round(relatedness * group_size)
        num_random = group_size - num_related

        if num_related > 0
            # Sample random individuals from the filtered candidates
            random_individuals = sample(candidates, num_random, replace=false)

            # Fill the group with related individuals and sampled individuals
            related_individuals = fill(focal_individual_index, num_related)
            final_group = [related_individuals; random_individuals]
        else
            # If relatedness is 0, just sample a full group
            random_individuals = sample(candidates, group_size - 1, replace=false)
            final_group = [focal_individual_index; random_individuals]
        end

        groups[i] = final_group
    end

    return groups
end

function collect_initial_conditions_and_parameters(groups::Vector{Vector{Int64}}, pop::Population)
    num_groups = pop.parameters.population_size
    group_size = pop.parameters.group_size

    u0s = Vector{SVector{group_size, Float32}}(undef, num_groups)
    ps = Vector{SVector{3 + group_size, Float32}}(undef, num_groups)

    for (i, group) in enumerate(groups)
        # Collect initial actions
        actions = [pop.action[idx] for idx in group]
        u0s[i] = SVector{group_size, Float32}(actions...)

        # Collect parameters
        int_pun_values = [pop.int_pun[idx] for idx in group]
        ps[i] = SVector{3 + group_size, Float32}(pop.norm_pool, pop.pun_pool, pop.parameters.synergy, int_pun_values...)
    end

    return u0s, ps
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{N, Float32}}, groups::Vector{Vector{Int64}}, pop::Population) where N
    for (group_indices, actions) in zip(groups, final_actions)
        # Update the action for each individual in the group
        for (i, idx) in enumerate(group_indices)
            pop.action[idx] = actions[i]
        end

        # Calculate and update payoffs for the group
        total_payoff!(group_indices, pop)
    end

    nothing
end

function social_interactions!(pop::Population)
    # Update norm and punishment pools
    update_norm_punishment_pools!(pop)

    # Shuffle and pair individuals
    groups= shuffle_and_group(pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    u0s, ps = collect_initial_conditions_and_parameters(groups, pop)
    final_actions = behav_eq(u0s, ps, pop.parameters.tmax, pop.parameters.population_size)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, groups, pop)

    nothing
end


##################
# Reproduction Function
##################

function reproduce!(pop::Population)
    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness(pop, i, pop.parameters.fitness_scaling_factor_a, pop.parameters.fitness_scaling_factor_b), 1:pop.parameters.population_size)
    # Create a list of indices corresponding to individuals
    indices_list = 1:pop.parameters.population_size

    # Sample indices with the given fitness weights
    sampled_indices = sample(indices_list, ProbabilityWeights(fitnesses), pop.parameters.population_size, replace=true, ordered=false)

    # Sort sampled indices to avoid unnecessary memory shuffling during offspring generation
    sort!(sampled_indices)

    # Create new offspring from sampled individuals
    for i in 1:pop.parameters.population_size
        offspring!(pop, i, sampled_indices[i])
    end

    nothing
end

#= Maximal fitness reproduction
function reproduce!(pop::Population)
    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness(pop, i, pop.parameters.fitness_scaling_factor_a, pop.parameters.fitness_scaling_factor_b), 1:pop.parameters.population_size)

    # Find the index of the individual with the highest fitness
    highest_fitness_index = argmax(fitnesses)

    # Get the traits of the highest fitness individual
    best_action = pop.action[highest_fitness_index]
    best_norm = pop.norm[highest_fitness_index]
    best_ext_pun = pop.ext_pun[highest_fitness_index]
    best_int_pun = pop.int_pun[highest_fitness_index]

    # Update population individuals based on maximal fitness
    for i in 1:pop.parameters.population_size
        pop.action[i] = best_action
        pop.norm[i] = best_norm
        pop.ext_pun[i] = best_ext_pun
        pop.int_pun[i] = best_int_pun
        pop.payoff[i] = 0.0      # Reset payoff for the new generation
        pop.interactions[i] = 0  # Reset interactions count
    end

    nothing
end
=#


##################
# Mutation Function 
##################

function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
    mutation_variance = pop.parameters.mutation_variance

    # Return immediately if no mutation is needed
    if mutation_variance == 0
        return nothing
    end

    mutation_rate = pop.parameters.mutation_rate
    lower_bound, upper_bound = truncate_bounds

    # Define distributions for mutation
    for i in 1:pop.parameters.population_size
        # Mutate `norm` trait
        if rand() <= mutation_rate
            norm_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -pop.norm[i]), upper=upper_bound)
            pop.norm[i] += rand(norm_dist)
        end

        # Mutate `ext_pun` trait
        if rand() <= mutation_rate
            ext_pun_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -pop.ext_pun[i]), upper=upper_bound)
            pop.ext_pun[i] += rand(ext_pun_dist)
        end

        # Uncomment to mutate `int_pun` as well
        # if rand() <= mutation_rate
        #     int_pun_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -pop.int_pun[i]), upper=upper_bound)
        #     pop.int_pun[i] += rand(int_pun_dist)
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

    # Iterate over each individual by index
    for i in 1:pop.parameters.population_size
        # Mutate `norm` trait
        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.norm[i] = max(0, pop.norm[i] + mutation_amount)
        end

        # Mutate `ext_pun` trait
        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.ext_pun[i] = max(0, pop.ext_pun[i] + mutation_amount)
        end

        # Uncomment to mutate `int_pun` as well
        # if rand() <= mutation_rate
        #     mutation_amount = rand(mutation_direction) * mutation_unit
        #     pop.int_pun[i] = max(0, pop.int_pun[i] + mutation_amount)
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