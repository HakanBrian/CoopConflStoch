using StatsBase, Random, Distributions, DataFrames, StaticArrays


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
    pop.int_pun_ext[offspring_index] = pop.int_pun_ext[parent_index]
    pop.int_pun_self[offspring_index] = pop.int_pun_self[parent_index]

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
    int_puns_ext = Vector{Float32}(undef, pop_size)
    int_puns_self = Vector{Float32}(undef, pop_size)
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
            int_puns_ext[i] = int_pun_ext0 + rand(int_pun_ext0_dist)
            int_puns_self[i] = int_pun_self0 + rand(int_pun_self0_dist)
        else
            actions[i] = action0
            norms[i] = norm0
            ext_puns[i] = ext_pun0
            int_puns_ext[i] = int_pun_ext0
            int_puns_self[i] = int_pun_self0
        end
        payoffs[i] = 0.0f0
        interactions[i] = 0
    end

    return Population(
        parameters,
        actions,
        norms,
        ext_puns,
        int_puns_ext,
        int_puns_self,
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
    outputs.T_ext[output_rows] = pop.int_pun_ext
    outputs.T_self[output_rows] = pop.int_pun_self
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

function total_payoff!(group::Vector{Int64}, pop::Population)
    group_norm = mean(@view pop.norm[group])
    group_pun = mean(@view pop.ext_pun[group])

    focal_idx = group[1] 

    # Extract the action of the focal individual as a real number (not a view)
    action_i = pop.action[focal_idx]

    # Collect actions from the other individuals in the group
    actions_j = @view pop.action[group[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, group_norm, group_pun, pop.parameters.synergy)

    # Update the individual's payoff and interactions
    pop.payoff[focal_idx] = (payoff_foc + pop.interactions[focal_idx] * pop.payoff[focal_idx]) / (pop.interactions[focal_idx] + 1)
    pop.interactions[focal_idx] += 1

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

function random_response(focal_idx::Int64, indices::Vector{Int64}, current_actions::AbstractVector{Float32}, pop::Population)
    synergy = pop.parameters.synergy
    exploration_rate = pop.parameters.exploration_rate

    # Get the group members' actions
    current_action = current_actions[focal_idx]
    group_actions = copy(current_actions)
    deleteat!(group_actions, focal_idx)

    # Get the internal norms
    int_pun_ext = pop.int_pun_ext[indices[focal_idx]]
    int_pun_self = pop.int_pun_self[indices[focal_idx]]

    # Compute norm_means
    norm_i = pop.norm[indices[focal_idx]]  # The i individual's norm
    norms = @view pop.norm[indices]
    norm_total = sum(norms)
    norm_total_mean = norm_total / length(norms)  # Mean of all norms in the group
    norm_others_mean = (norm_total - norm_i) / (length(norms) - 1)  # The -i individuals' mean norm

    # Compute the mean of external punishments
    pun_mean = mean(@view pop.ext_pun[indices])

    # Calculate current payoff for the individual
    current_payoff = objective(current_action, group_actions,
                               norm_i,
                               norm_others_mean,
                               norm_total_mean,
                               pun_mean,
                               int_pun_ext,
                               int_pun_self,
                               synergy)

    # Exploration: Add randomness to encourage exploring new strategies
    exploration = exploration_rate * (2rand() - 1)
    best_action = current_action
    new_action = current_action + exploration

    # Compute payoff with the new action
    new_payoff = objective(new_action, group_actions,
                            norm_i,
                            norm_others_mean,
                            norm_total_mean,
                            pun_mean,
                            int_pun_ext,
                            int_pun_self,
                            synergy)

    # Update the action if the new payoff is better
    if new_payoff > current_payoff
        best_action = new_action
    end

    return best_action
end

function behavioral_equilibrium!(indices::Vector{Int64}, pop::Population)
    tolerance = pop.parameters.tolerance
    group_tolerance = 0.001

    max_time_steps = pop.parameters.max_time_steps
    time_step = 0

    current_actions = @view pop.action[indices]

    while time_step < max_time_steps
        time_step += 1
        action_change = 0.0  # Reset action_change for each iteration

        # Track the sum of squared differences from the group's mean action
        group_mean = mean(current_actions)
        group_stability = 0.0

        for i in eachindex(indices)
            best_action = random_response(i, indices, current_actions, pop)
            action_change = max(action_change, abs(best_action - current_actions[i]))

            # Calculate group stability: sum of squared differences from the group's mean action
            group_stability += (best_action - group_mean)^2

            # Update action in the population
            pop.action[indices[i]] = best_action
        end

        # Normalize group stability by the group size
        group_stability /= pop.parameters.group_size

        # Check if the action change and group stability are both below their respective thresholds
        if action_change < tolerance && group_stability < group_tolerance
            pop.action[indices] .= mean(pop.action[indices])
            break
        end
    end

    nothing
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
    
    # Create a matrix with `population_size` rows and `group_size` columns
    groups = Matrix{Int64}(undef, population_size, group_size)

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

        # Assign the final group to the matrix row
        groups[i, :] = final_group
    end

    return groups
end

function social_interactions!(pop::Population)
    # Shuffle and pair individuals
    groups = shuffle_and_group(pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Calculate equilibrium actions for all pairs
    for i in axes(groups, 1)
        group = groups[i, :]
        behavioral_equilibrium!(group, pop)
    end

    for i in axes(groups, 1)
        group = groups[i, :]
        total_payoff!(group, pop)
    end
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

        # Mutate `int_pun_ext` trait
        if rand() <= mutation_rate
            int_pun_ext_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -pop.int_pun_ext[i]), upper=upper_bound)
            pop.int_pun_ext[i] += rand(int_pun_ext_dist)
        end
        
        # Mutate `int_pun_self` trait
        if rand() <= mutation_rate
            int_pun_self_dist = truncated(Normal(0, mutation_variance), lower=max(lower_bound, -pop.int_pun_self[i]), upper=upper_bound)
            pop.int_pun_self[i] += rand(int_pun_self_dist)
        end
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

        # Mutate `int_pun_ext` trait
        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_ext[i] = max(0, pop.int_pun_ext[i] + mutation_amount)
        end

        # Mutate `int_pun_self` trait
        if rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_self[i] = max(0, pop.int_pun_self[i] + mutation_amount)
        end
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

    output_length = floor(Int64, pop.parameters.generations / pop.parameters.output_save_tick) * pop.parameters.population_size
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

    for t in 1:pop.parameters.generations
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