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
        interactions
    )
end

function output!(outputs::DataFrame, t::Int64, pop::Population)
    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * pop.parameters.population_size + 1
    end

    N = pop.parameters.population_size

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base + N - 1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = fill(t, N)
    outputs.individual[output_rows] = 1:N
    outputs.action[output_rows] = pop.action
    outputs.a[output_rows] = pop.norm
    outputs.p[output_rows] = pop.ext_pun
    outputs.T[output_rows] = pop.int_pun
    outputs.payoff[output_rows] = pop.payoff

    nothing
end


##################
# Fitness Function
##################

function benefit(action_i::Real, actions_j::AbstractVector{<:Real}, synergy::Real)
    sqrt_action_i = sqrt(action_i)
    sum_sqrt_actions_j = mapreduce(sqrt, +, actions_j)
    sum_sqrt_actions = sqrt_action_i + sum_sqrt_actions_j
    sqrt_sum_actions = sqrt(action_i + sum(actions_j))
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

function payoff(action_i::Real, actions_j::AbstractVector{<:Real}, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(action_i, actions_j, synergy) - cost(action_i) - external_punishment(action_i, norm_pool, punishment_pool)
end

function objective(action_i::Real, actions_j::AbstractVector{<:Real}, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return payoff(action_i, actions_j, norm_pool, punishment_pool, synergy) - internal_punishment(action_i, norm_pool, T)
end

function objective_derivative(action_i::Real, actions_j::AbstractVector{<:Real}, norm_pool::Real, punishment_pool::Real, T::Real, synergy::Real)
    return ForwardDiff.derivative(x -> objective(x, actions_j, norm_pool, punishment_pool, T, synergy), action_i)
end

function total_payoff!(group_indices::Vector{Int64}, group_pool::Vector{Float32}, pop::Population)
    # Extract the action of the focal individual
    action_i = pop.action[group_indices[1]]

    # Collect actions from the other individuals in the group
    actions_j = pop.action[group_indices[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, group_pool[1], group_pool[2], pop.parameters.synergy)

    # Update the individual's payoff and interactions
    idx = group_indices[1]
    pop.payoff[idx] = (payoff_foc + pop.interactions[idx] * pop.payoff[idx]) / (pop.interactions[idx] + 1)
    pop.interactions[idx] += 1

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

function behav_eq(u0s::Matrix{Float32}, ps::Matrix{Float32}, group_pools::Matrix{Float32}, synergy::Float64, tmax::Float64, num_groups::Int64, group_size::Int64)
    u0 = SVector{group_size, Float32}(u0s[:, 1])
    tspan = Float32[0.0, tmax]
    p = SVector{3 + group_size, Float32}(vcat(group_pools[1, 1], group_pools[2, 1], synergy, ps[:, 1]))

    # Create an initial problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0, tspan, p)

    # Function to remake the problem for each group
    prob_func = (prob, i, repeat) -> remake(prob,
                                            u0 = SVector{group_size, Float32}(u0s[:, i]),
                                            p = SVector{3 + group_size, Float32}(vcat(group_pools[1, i], group_pools[2, i], synergy, ps[:, i])))

    # Create an ensemble problem, configured for GPU execution
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem using a GPU kernel
    sim = solve(ensemble_prob, EnsembleGPUKernel(CUDA.CUDABackend()), trajectories=num_groups, save_start=false, save_on=false)

    # Extract final action values
    final_actions = [sol[end] for sol in sim]

    return final_actions
end


##################
# Social Interactions Function
##################

function probabilistic_round(x::Float64)::Int64
    lower = floor(Int64, x)
    upper = ceil(Int64, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

function shuffle_and_group(population_size::Int64, group_size::Int64, relatedness::Float64)
    individuals_indices = collect(1:population_size)
    shuffle!(individuals_indices)
    groups = Vector{Vector{Int64}}(undef, population_size)

    # Iterate over each individual index and form a group
    for i in 1:population_size
        focal_individual_index = individuals_indices[i]

        # Create a list of potential candidates excluding the focal individual
        candidates = filter(x -> x != focal_individual_index, individuals_indices)

        # Calculate the number of related individuals using probabilistic rounding
        num_related = probabilistic_round(relatedness * group_size)
        num_random = group_size - num_related

        if num_related > 0
            # Sample random individuals from the filtered candidates
            random_individuals = sample(candidates, num_random, replace=false)

            # Fill the group with related individuals and sampled individuals
            related_individuals = fill(focal_individual_index, num_related)
            final_group = vcat(related_individuals, random_individuals)
        else
            # If relatedness is 0, just sample a full group
            random_individuals = sample(candidates, group_size - 1, replace=false)
            final_group = vcat(focal_individual_index, random_individuals)
        end

        groups[i] = final_group
    end

    return groups
end

function collect_initial_conditions_and_parameters(groups::Vector{Vector{Int64}}, pop::Population)
    num_groups = pop.parameters.population_size
    group_size = pop.parameters.group_size

    # Initialize u0s for action values and ps for individual punishments
    u0s = zeros(Float32, group_size, num_groups)
    ps = zeros(Float32, group_size, num_groups)

    action_values = Vector{Float32}(undef, group_size)
    norm_values = Vector{Float32}(undef, group_size)
    int_pun_values = Vector{Float32}(undef, group_size)
    ext_pun_values = Vector{Float32}(undef, group_size)

    group_pools = zeros(Float32, 2, num_groups)  # 2 rows: norm_pools, pun_pools

    # Collect initial conditions and parameters for each group
    for (i, group) in enumerate(groups)
        for j in 1:group_size
            member_index = group[j]
            action_values[j] = pop.action[member_index]
            norm_values[j] = pop.norm[member_index]
            ext_pun_values[j] = pop.ext_pun[member_index]
            int_pun_values[j] = pop.int_pun[member_index]
        end

        # Calculate the pools and store them in the matrix
        group_pools[1, i] = mean(norm_values)  # norm_pools
        group_pools[2, i] = mean(ext_pun_values)  # pun_pools

        # Assign values to matrices
        u0s[:, i] = action_values
        ps[:, i] = int_pun_values
    end

    return u0s, ps, group_pools
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{N, Float32}}, groups::Vector{Vector{Int64}}, group_pools::Matrix{Float32}, pop::Population) where N
    # Transpose the matrix to access columns as rows
    transposed_pools = group_pools'

    for (actions, group_indices, group_pool) in zip(final_actions, groups, eachrow(transposed_pools))
        # Update the action for each individual in the group
        for (i, idx) in enumerate(group_indices)
            pop.action[idx] = actions[i]
        end

        # Calculate and update payoffs for the group
        total_payoff!(group_indices, collect(group_pool), pop)
    end

    nothing
end

function social_interactions!(pop::Population)
    # Shuffle and pair individuals
    groups = shuffle_and_group(pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    u0s, ps, group_pools = collect_initial_conditions_and_parameters(groups, pop)
    final_actions = behav_eq(u0s, ps, group_pools, pop.parameters.synergy, pop.parameters.tmax, pop.parameters.population_size, pop.parameters.group_size)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, groups, group_pools, pop)

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

    # Update population individuals based on maximal fitness
    for i in 1:pop.parameters.population_size
        offspring!(pop, i, highest_fitness_index)
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