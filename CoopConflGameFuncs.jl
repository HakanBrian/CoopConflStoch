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
    pop.int_pun[:, offspring_index] = pop.int_pun[:, parent_index]

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
    int_pun_ext0 = parameters.int_pun_ext0
    int_pun_self0 = parameters.int_pun_self0
    pop_size = parameters.population_size

    # Initialize arrays for attributes
    actions = Vector{Float32}(undef, pop_size)
    norms = Vector{Float32}(undef, pop_size)
    ext_puns = Vector{Float32}(undef, pop_size)
    int_puns = zeros(Float32, 2, pop_size)
    payoffs = Vector{Float32}(undef, pop_size)
    interactions = Vector{Int64}(undef, pop_size)

    # Construct distributions if necessary
    if use_distribution
        lower_bound, upper_bound = truncation_bounds(trait_variance, 0.99)
        action0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -action0), upper=upper_bound)
        norm0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -norm0), upper=upper_bound)
        ext_pun0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -ext_pun0), upper=upper_bound)
        int_pun_ext0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -int_pun_ext0), upper=upper_bound)
        int_pun_self0_dist = truncated(Normal(0, trait_variance), lower=max(lower_bound, -int_pun_self0), upper=upper_bound)
    end

    # Create individuals
    for i in 1:pop_size
        if use_distribution
            actions[i] = action0 + rand(action0_dist)
            norms[i] = norm0 + rand(norm0_dist)
            ext_puns[i] = ext_pun0 + rand(ext_pun0_dist)
            int_puns[1, i] = int_pun_ext0 + rand(int_pun_ext0_dist)
            int_puns[2, i] = int_pun_self0 + rand(int_pun_self0_dist)
        else
            actions[i] = action0
            norms[i] = norm0
            ext_puns[i] = ext_pun0
            int_puns[1, i] = int_pun_ext0
            int_puns[2, i] = int_pun_self0
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
    N = pop.parameters.population_size

    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * N + 1
    end

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base + N - 1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = fill(t, N)
    outputs.individual[output_rows] = 1:N
    outputs.action[output_rows] = pop.action
    outputs.a[output_rows] = pop.norm
    outputs.p[output_rows] = pop.ext_pun
    outputs.T_ext[output_rows] = pop.int_pun[1, :]
    outputs.T_self[output_rows] = pop.int_pun[2, :]
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

function cost(action_i::Real)
    return action_i^2
end

function external_punishment(action_i::Real, norm_pool::Real, punishment_pool::Real)
    return punishment_pool * (action_i - norm_pool)^2
end

function internal_punishment_ext(action_i::Real, norm_pool_mini::Real, T_ext::Real)
    return T_ext * (action_i - norm_pool_mini)^2
end

function internal_punishment_self(action_i::Real, norm_i::Real, T_self::Real)
    return T_self * (action_i - norm_i)^2
end

function payoff(action_i::Real, actions_j::AbstractVector{<:Real}, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(action_i, actions_j, synergy) - cost(action_i) - external_punishment(action_i, norm_pool, punishment_pool)
end

function objective(action_i::Real, actions_j::AbstractVector{<:Real}, norm_i::Real, norm_mini::Real, norm_pool::Real, punishment_pool::Real, T_ext::Real, T_self::Real, synergy::Real)
    return payoff(action_i, actions_j, norm_pool, punishment_pool, synergy) - internal_punishment_ext(action_i, norm_mini, T_ext) - internal_punishment_self(action_i, norm_i, T_self)
end

function objective_derivative(action_i::Real, actions_j::AbstractVector{<:Real}, norm_i::Real, norm_mini::Real, norm_pool::Real, punishment_pool::Real, T_ext::Real, T_self::Real, synergy::Real)
    return ForwardDiff.derivative(x -> objective(x, actions_j, norm_i, norm_mini, norm_pool, punishment_pool, T_ext, T_self, synergy), action_i)
end

function total_payoff!(group_indices::Vector{Int64}, group_norm::Float32, group_pun::Float32, pop::Population)
    # Extract the action of the focal individual
    action_i = pop.action[group_indices[1]]

    # Collect actions from the other individuals in the group
    actions_j = pop.action[group_indices[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, group_norm, group_pun, pop.parameters.synergy)

    # Update the individual's payoff and interactions
    idx = group_indices[1]
    pop.payoff[idx] = (payoff_foc + pop.interactions[idx] * pop.payoff[idx]) / (pop.interactions[idx] + 1)
    pop.interactions[idx] += 1

    nothing
end

function fitness(pop::Population, idx::Int64)
    return pop.payoff[idx] - pop.ext_pun[idx]
end

function fitness(pop::Population, idx::Int64, fitness_scaling_factor_a::Float64, fitness_scaling_factor_b::Float64)
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
    du = ntuple(i -> objective_derivative(u[i], remove_element(u, i), p[1], p[2], p[3], p[4], p[5 + i], p[5 + i + N], p[5]), N)
    return SVector{N}(du)
end

function behav_eq(action0s::Matrix{Float32}, T_ext::Matrix{Float32}, T_self::Matrix{Float32}, group_norm_pools::Matrix{Float32}, group_pun_pools::Matrix{Float32}, synergy::Float64, tmax::Float64, num_groups::Int64, group_size::Int64)
    u0 = SVector{group_size, Float32}(action0s[:, 1])
    tspan = Float32[0.0f0, tmax]
    p = SVector{5 + 2*group_size, Float32}(vcat(group_norm_pools[:, 1][1], mean(group_norm_pools[:, 1][2:end]), mean(group_norm_pools[:, 1]), mean(group_pun_pools[:, 1]), synergy, T_ext[:, 1], T_self[:, 1]))

    # Create an initial problem with the first set of parameters as a template
    prob = ODEProblem{false}(behav_ODE_static, u0, tspan, p)

    # Function to remake the problem for each group
    prob_func = (prob, i, repeat) -> remake(prob,
                                            u0 = SVector{group_size, Float32}(action0s[:, i]),
                                            p = SVector{5 + 2*group_size, Float32}(vcat(group_norm_pools[:, i][1], mean(group_norm_pools[:, i][2:end]), mean(group_norm_pools[:, i]), mean(group_pun_pools[:, i]), synergy, T_ext[:, i], T_self[:, i])))

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

    action0s = zeros(Float32, group_size, num_groups)

    group_norm_pools = zeros(Float32, group_size, num_groups)
    group_pun_pools = zeros(Float32, group_size, num_groups)

    int_pun_ext = zeros(Float32, group_size, num_groups)
    int_pun_self = zeros(Float32, group_size, num_groups)

    for (i, group) in enumerate(groups)
        for j in 1:group_size
            member_index = group[j]
            action0s[j, i] = pop.action[member_index]
            group_norm_pools[j, i] = pop.norm[member_index]
            group_pun_pools[j, i] = pop.ext_pun[member_index]
            int_pun_ext[j, i] = pop.int_pun[1, member_index]
            int_pun_self[j, i] = pop.int_pun[2, member_index]
        end
    end

    return action0s, int_pun_ext, int_pun_self, group_norm_pools, group_pun_pools
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{N, Float32}}, groups::Vector{Vector{Int64}}, group_norm_pools::Matrix{Float32}, group_pun_pools::Matrix{Float32}, pop::Population) where N
    for (j, actions, group_indices) in zip(1:pop.parameters.population_size, final_actions, groups)
        # Update the action for each individual in the group
        for (i, idx) in enumerate(group_indices)
            pop.action[idx] = actions[i]
        end

        # Calculate and update payoffs for the group
        total_payoff!(group_indices, mean(group_norm_pools[:, j]), mean(group_pun_pools[:, j]), pop)
    end

    nothing
end

function social_interactions!(pop::Population)
    # Shuffle and pair individuals
    groups = shuffle_and_group(pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    action0s, int_pun_ext, int_pun_self, group_norm_pools, group_pun_pools = collect_initial_conditions_and_parameters(groups, pop)
    final_actions = behav_eq(action0s, int_pun_ext, int_pun_self, group_norm_pools, group_pun_pools, pop.parameters.synergy, pop.parameters.tmax, pop.parameters.population_size, pop.parameters.group_size)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, groups, group_norm_pools, group_pun_pools, pop)

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
        T_ext = Vector{Float64}(undef, output_length),
        T_self = Vector{Float64}(undef, output_length),
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