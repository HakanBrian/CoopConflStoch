using Core.Intrinsics, StatsBase, Random, Distributions, DataFrames, StaticArrays, Distributed


####################################
# Helper Functions
####################################

include("CoopConflGameStructs.jl")
include("CoopConflGameHelper.jl")


###############################
# Population Simulation
###############################

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
    group_size = parameters.group_size

    # Initialize arrays for attributes
    actions = Vector{Float32}(undef, pop_size)
    norms = Vector{Float32}(undef, pop_size)
    ext_puns = Vector{Float32}(undef, pop_size)
    int_puns_ext = Vector{Float32}(undef, pop_size)
    int_puns_self = Vector{Float32}(undef, pop_size)
    payoffs = Vector{Float32}(undef, pop_size)
    interactions = Vector{Int64}(undef, pop_size)
    groups = Matrix{Int64}(undef, pop_size, group_size)

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
        interactions,
        groups
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
# Cost and Punishemnt
##################

@inline function cost(action_i::Float32)
    return action_i^2
end

@inline function external_punishment(action_i::Float32, norm_pool::Float32, punishment_pool::Float32)
    return punishment_pool * (action_i - norm_pool)^2
end

@inline function internal_punishment_I(action_i::Float32, norm_pool::Float32, T_ext::Float32)
    return T_ext * (action_i - norm_pool)^2
end

@inline function internal_punishment_II(action_i::Float32, norm_pool::Float32, T_ext::Float32)
    return T_ext * log(1 + ((action_i - norm_pool)^2))
end

@inline function internal_punishment_ext(action_i::Float32, norm_pool_mini::Float32, T_ext::Float32)
    return T_ext * (action_i - norm_pool_mini)^2
end

@inline function internal_punishment_self(action_i::Float32, norm_i::Float32, T_self::Float32)
    return T_self * (action_i - norm_i)^2
end


##################
# Benefit
##################

@inline function benefit(action_i::Float32, actions_j::AbstractVector{Float32})
    sqrt_action_i = sqrt_llvm(action_i)
    sum_sqrt_actions_j = sum_sqrt_loop(actions_j)

    return sqrt_action_i + sum_sqrt_actions_j
end

@inline function benefit(action_i::Float32, actions_j::AbstractVector{Float32}, synergy::Float32)
    sqrt_action_i = sqrt_llvm(action_i)
    sum_sqrt_actions_j = sum_sqrt_loop(actions_j)
    sum_sqrt_actions = sqrt_action_i + sum_sqrt_actions_j
    sqrt_sum_actions = sqrt_sum_loop(action_i, ctions_j)

    return (1 - synergy) * sum_sqrt_actions + synergy * sqrt_sum_actions
end

@inline function benefit_sqrt(action_i::Float32, actions_j::Float32)
    return action_i + actions_j
end

##################
# payoff and objective
##################

# Normal version =================================
@inline function payoff(action_i::Float32, actions_j::AbstractVector{Float32}, norm_pool::Float32, punishment_pool::Float32)
    b = benefit(action_i, actions_j)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(action_i::Float32, actions_j::AbstractVector{Float32}, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32)
    p = payoff(action_i, actions_j, norm_pool, punishment_pool)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(action_i::Float32, actions_j::AbstractVector{Float32}, norm_i::Float32, norm_mini::Float32, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32, T_self::Float32)
    p = payoff(action_i, actions_j, norm_pool, punishment_pool)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

# Normal Sqrt version =================================
@inline function payoff(action_i::Float32, action_i_sqrt::Float32, actions_j::Float32, norm_pool::Float32, punishment_pool::Float32)
    b = benefit_sqrt(action_i_sqrt, actions_j)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(action_i::Float32, action_i_sqrt::Float32, actions_j::Float32, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32)
    p = payoff(action_i, action_i_sqrt, actions_j, norm_pool, punishment_pool)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(action_i::Float32, action_i_sqrt::Float32, actions_j::Float32, norm_i::Float32, norm_mini::Float32, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32, T_self::Float32)
    p = payoff(action_i, action_i_sqrt, actions_j, norm_pool, punishment_pool)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

# Synergy version =================================
@inline function payoff(action_i::Float32, actions_j::AbstractVector{Float32}, norm_pool::Float32, punishment_pool::Float32, synergy::Float32)
    b = benefit(action_i, actions_j, synergy)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(action_i::Float32, actions_j::AbstractVector{Float32}, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32, synergy::Float32)
    p = payoff(action_i, actions_j, norm_pool, punishment_pool, synergy)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(action_i::Float32, actions_j::AbstractVector{Float32}, norm_i::Float32, norm_mini::Float32, norm_pool::Float32, punishment_pool::Float32, T_ext::Float32, T_self::Float32, synergy::Float32)
    p = payoff(action_i, actions_j, norm_pool, punishment_pool, synergy)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

function total_payoff!(group::AbstractVector{Int64}, norm_pool::Float32, pun_pool::Float32, pop::Population)
    focal_idiv = group[1]

    # Extract the action of the focal individual as a real number (not a view)
    action_i = @inbounds pop.action[focal_idiv]

    # Collect actions from the other individuals in the group
    actions_j = @inbounds @view pop.action[@view group[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, norm_pool, pun_pool)

    # Update the individual's payoff and interactions
    pop.payoff[focal_idiv] = (payoff_foc + pop.interactions[focal_idiv] * pop.payoff[focal_idiv]) / (pop.interactions[focal_idiv] + 1)
    pop.interactions[focal_idiv] += 1

    nothing
end


##################
# Fitness
##################

@inline function fitness(pop::Population, idx::Int64)
    return pop.payoff[idx] - pop.ext_pun[idx]
end

function fitness_exp(pop::Population, idx::Int64)
    base_fitness = fitness(pop, idx)
    return exp(base_fitness * 10.0)
end

function fitness_exp_norm(pop::Population, idx::Int64)
    base_fitness = fitness(pop, idx)
    return Exponential(base_fitness * 10.0)
end


##################
# Behavioral Equilibrium
##################

@inline function best_response(focal_idx::Int64, group::AbstractVector{Int64}, action_sqrt_view::AbstractVector{Float32}, action_sqrt_sum::Float32, norm_pool::Float32, pun_pool::Float32, pop::Population, delta_action::Float32)
    group_size = pop.parameters.group_size
    focal_indiv = @inbounds group[focal_idx]

    # Get the actions
    action_i = @inbounds pop.action[focal_indiv]
    action_i_sqrt = @inbounds action_sqrt_view[focal_idx]
    action_j_filtered_view_sum = action_sqrt_sum - action_i_sqrt

    # Get the internal punishments
    int_pun_ext = @inbounds pop.int_pun_ext[focal_indiv]
    int_pun_self = @inbounds pop.int_pun_self[focal_indiv]

    # Compute norm means
    norm_i = @inbounds pop.norm[focal_indiv]  # Norm of i individual
    norm_mini = (norm_pool * group_size - norm_i) / (group_size - 1)  # Mean norm of -i individuals

    # Calculate current payoff for the individual
    current_payoff = objective(action_i, 
                            action_i_sqrt,
                            action_j_filtered_view_sum,
                            norm_i,
                            norm_mini,
                            norm_pool,
                            pun_pool,
                            int_pun_ext,
                            int_pun_self)

    # Perturb action upwards
    action_up = action_i + delta_action
    action_up_sqrt = sqrt_llvm(action_up)

    # Calculate new payoffs with perturbed actions
    new_payoff_up = objective(action_up, 
                            action_up_sqrt,
                            action_j_filtered_view_sum,
                            norm_i,
                            norm_mini,
                            norm_pool,
                            pun_pool,
                            int_pun_ext,
                            int_pun_self)

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_up > current_payoff
        return action_up, action_up_sqrt
    end

    # Perturb action downwards
    action_down = max(action_i - delta_action, 0.0f0)
    action_down_sqrt = sqrt_llvm(action_down)

    # Calculate new payoffs with perturbed actions
    new_payoff_down = objective(action_down, 
                            action_down_sqrt,
                            action_j_filtered_view_sum,
                            norm_i,
                            norm_mini,
                            norm_pool,
                            pun_pool,
                            int_pun_ext,
                            int_pun_self)

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_down > current_payoff
        return action_down, action_down_sqrt
    end

    return action_i, action_i_sqrt
end

#= Single internal punishment
@inline function best_response(focal_idx::Int64, group::AbstractVector{Int64}, action_sqrt_view::AbstractVector{Float32}, action_sqrt_sum::Float32, norm_pool::Float32, pun_pool::Float32, pop::Population, delta_action::Float32)
    focal_indiv = @inbounds group[focal_idx]

    # Get the actions
    action_i = @inbounds pop.action[focal_indiv]
    action_i_sqrt = @inbounds action_sqrt_view[focal_idx]
    action_j_filtered_view_sum = action_sqrt_sum - action_i_sqrt

    # Get the internal punishments
    int_pun_ext = @inbounds pop.int_pun_ext[focal_indiv]

    # Calculate current payoff for the individual
    current_payoff = objective(action_i, 
                            action_i_sqrt,
                            action_j_filtered_view_sum,
                            norm_pool,
                            pun_pool,
                            int_pun_ext)

    # Perturb action upwards
    action_up = action_i + delta_action
    action_up_sqrt = sqrt_llvm(action_up)

    # Calculate new payoffs with perturbed actions
    new_payoff_up = objective(action_up, 
                            action_up_sqrt,
                            action_j_filtered_view_sum,
                            norm_pool,
                            pun_pool,
                            int_pun_ext)

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_up > current_payoff
        return action_up, action_up_sqrt
    end

    # Perturb action downwards
    action_down = max(action_i - delta_action, 0.0f0)
    action_down_sqrt = sqrt_llvm(action_down)

    # Calculate new payoffs with perturbed actions
    new_payoff_down = objective(action_down, 
                            action_down_sqrt,
                            action_j_filtered_view_sum,
                            norm_pool,
                            pun_pool,
                            int_pun_ext)

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_down > current_payoff
        return action_down, action_down_sqrt
    end

    return action_i, action_i_sqrt
end
=#

function behavioral_equilibrium!(group::AbstractVector{Int64}, action_sqrt::Vector{Float32}, action_sqrt_sum::Float32, norm_pool::Float32, pun_pool::Float32, pop::Population)
    # Collect parameters
    tolerance = pop.parameters.tolerance
    max_time_steps = pop.parameters.max_time_steps

    # Create a view for group actions
    temp_actions = @inbounds view(pop.action, group)
    action_sqrt_view = @inbounds view(action_sqrt, group)

    action_change = 1.0f0
    delta_action = 0.1f0
    time_step = 0
    while time_step < max_time_steps
        time_step += 1

        # Dynamically adjust delta_action
        if delta_action < tolerance
            break
        elseif action_change == 0.0f0
            delta_action *= 0.5f0
        else
            delta_action *= 1.5f0
        end

        action_change = 0.0f0  # Reset action_change for each iteration

        # Calculate the relatively best action of each individual in the group
        for i in eachindex(group)
            best_action, best_action_sqrt = best_response(i, group, action_sqrt_view, action_sqrt_sum, norm_pool, pun_pool, pop, delta_action)
            diff = abs(best_action - temp_actions[i])
            if diff > action_change
                action_change = diff
            end
            temp_actions[i] = best_action
            action_sqrt_sum -= action_sqrt_view[i]
            action_sqrt_view[i] = best_action_sqrt
            action_sqrt_sum += best_action_sqrt
        end

    end

    nothing
end


##################
# Social Interaction
##################

function shuffle_and_group(groups::Matrix{Int64}, population_size::Int64, group_size::Int64, relatedness::Float64)
    individuals_indices = collect(1:population_size)
    shuffle!(individuals_indices)

    # Pre-allocate a buffer for candidates (one less than population size, since focal individual is excluded)
    candidates_buffer = Vector{Int64}(undef, population_size - 1)

    # Iterate over each individual index and form a group
    for i in 1:population_size
        focal_individual_index = individuals_indices[i]

        # Filter out the focal individual
        candidates_filtered_view = filter_out_val!(individuals_indices, focal_individual_index, candidates_buffer)

        # Calculate the number of related and random individuals
        num_related = probabilistic_round(relatedness * (group_size - 1))
        num_random = group_size - num_related - 1

        # Assign the focal individual to the group
        groups[i, :] .= focal_individual_index

        # Assign random individuals to the group
        groups[i, end-num_random+1:end] = in_place_sample!(candidates_filtered_view, num_random)
    end

    return groups
end

function find_actions_payoffs!(final_actions::Vector{Float32}, action_sqrt::Vector{Float32}, groups::Matrix{Int64}, pop::Population)
    # Iterate over each group to find actions and payoffs
    for i in axes(groups, 1)
        group = @inbounds @view groups[i, :]

        norm_pool = 0.0f0
        pun_pool = 0.0f0
        action_sqrt_sum = 0.0f0
        for member in group
            @inbounds norm_pool += pop.norm[member]
            @inbounds pun_pool += pop.ext_pun[member]
            @inbounds action_sqrt_sum += action_sqrt[member]
        end
        norm_pool /= pop.parameters.group_size
        pun_pool /= pop.parameters.group_size

        # Calculate equilibrium actions then payoffs for current groups
        behavioral_equilibrium!(group, action_sqrt, action_sqrt_sum, norm_pool, pun_pool, pop)
        total_payoff!(group, norm_pool, pun_pool, pop)

        # Update final actions
        final_actions[group[1]] = pop.action[group[1]]
    end
end

function social_interactions!(pop::Population)
    # Pre-allocate vectors
    final_actions = Vector{Float32}(undef, pop.parameters.population_size)
    action_sqrt = Vector{Float32}(undef, pop.parameters.population_size)

    # Cache the square root of actions
    action_sqrt = map(action -> sqrt_llvm(action), pop.action)

    # Shuffle and group individuals
    groups = shuffle_and_group(pop.groups, pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Get actions while updating payoff
    find_actions_payoffs!(final_actions, action_sqrt, groups, pop)

    # Update the population values with the equilibrium actions
    pop.action = final_actions

    nothing
end


##################
# Reproduction
##################

function reproduce!(pop::Population)
    # Create a list of indices corresponding to individuals
    indices_list = 1:pop.parameters.population_size

    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness_exp_norm(pop, i), indices_list)

    # Sample indices with the given fitness weights
    normalized_probs = normalize_exponentials(fitnesses)
    sampled_indices = sample(indices_list, Weights(normalized_probs), pop.parameters.population_size, replace=true, ordered=false)

    # Sort sampled indices to avoid unnecessary memory shuffling during offspring generation
    sort!(sampled_indices)

    # Create new offspring from sampled individuals
    for i in 1:pop.parameters.population_size
        offspring!(pop, i, sampled_indices[i])
    end

    nothing
end

#= Nonsybolic version
function reproduce!(pop::Population)
    # Create a list of indices corresponding to individuals
    indices_list = 1:pop.parameters.population_size

    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness_exp(pop, i), indices_list)

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
=#

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
# Mutation 
##################

function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
    mutation_variance = pop.parameters.mutation_variance

    # Return immediately if no mutation is needed
    if mutation_variance == 0
        return nothing
    end

    mutation_rate = pop.parameters.mutation_rate
    lower_bound, upper_bound = truncate_bounds
    mutation_dist = Normal(0, mutation_variance)

    # Define distributions for mutation
    for i in 1:pop.parameters.population_size
        # Mutate `norm` trait
        if pop.parameters.norm_mutation_enabled && rand() <= mutation_rate
            norm_dist = truncated(mutation_dist, lower=max(lower_bound, -pop.norm[i]), upper=upper_bound)
            pop.norm[i] += rand(norm_dist)
        end

        # Mutate `ext_pun` trait
        if pop.parameters.ext_pun_mutation_enabled && rand() <= mutation_rate
            ext_pun_dist = truncated(mutation_dist, lower=max(lower_bound, -pop.ext_pun[i]), upper=upper_bound)
            pop.ext_pun[i] += rand(ext_pun_dist)
        end

        # Mutate `int_pun_ext` trait
        if pop.parameters.int_pun_ext_mutation_enabled && rand() <= mutation_rate
            int_pun_ext_dist = truncated(mutation_dist, lower=max(lower_bound, -pop.int_pun_ext[i]), upper=upper_bound)
            pop.int_pun_ext[i] += rand(int_pun_ext_dist)
        end

        # Mutate `int_pun_self` trait
        if pop.parameters.int_pun_self_mutation_enabled && rand() <= mutation_rate
            int_pun_self_dist = truncated(mutation_dist, lower=max(lower_bound, -pop.int_pun_self[i]), upper=upper_bound)
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
        if pop.parameters.norm_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.norm[i] = max(0, pop.norm[i] + mutation_amount)
        end

        # Mutate `ext_pun` trait
        if pop.parameters.ext_pun_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.ext_pun[i] = max(0, pop.ext_pun[i] + mutation_amount)
        end

        # Mutate `int_pun_ext` trait
        if pop.parameters.int_pun_ext_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_ext[i] = max(0, pop.int_pun_ext[i] + mutation_amount)
        end

        # Mutate `int_pun_self` trait
        if pop.parameters.int_pun_self_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_self[i] = max(0, pop.int_pun_self[i] + mutation_amount)
        end
    end

    nothing
end
=#


#######################
# Simulation #
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

function run_simulation(parameters::SimulationParameters, param_id::Int64, replicate_id::Int64)
    println("Running simulation replicate $replicate_id for param_id $param_id")

    # Run the simulation
    population = population_construction(parameters)
    simulation_replicate = simulation(population)

    # Group by generation and compute mean for each generation
    simulation_gdf = groupby(simulation_replicate, :generation)
    simulation_mean = combine(simulation_gdf,
                                 :action => mean,
                                 :a => mean,
                                 :p => mean,
                                 :T_ext => mean,
                                 :T_self => mean,
                                 :payoff => mean)

    # Add columns for replicate and param_id
    rows_to_insert = nrow(simulation_mean)
    insertcols!(simulation_mean, 1, :param_id => fill(param_id, rows_to_insert))
    insertcols!(simulation_mean, 2, :replicate => fill(replicate_id, rows_to_insert))

    return simulation_mean
end

function simulation_replicate(parameters::SimulationParameters, num_replicates::Int64)
    # Use pmap to parallelize the simulation
    results = pmap(1:num_replicates) do i
        run_simulation(parameters, 1, i)
    end

    # Concatenate all the simulation means returned by each worker
    all_simulation_means = vcat(results...)

    return all_simulation_means
end

function simulation_replicate(parameter_sweep::Vector{SimulationParameters}, num_replicates::Int64)
    # Create a list of tasks (parameter set index, parameter set, replicate) to distribute
    tasks = [(idx, parameters, replicate) for (idx, parameters) in enumerate(parameter_sweep) for replicate in 1:num_replicates]

    # Use pmap to distribute the tasks across the workers
    results = pmap(tasks) do task
        param_idx, parameters, replicate = task
        # Run simulation and store the result with the parameter set index
        run_simulation(parameters, param_idx, replicate)
    end

    # Concatenate all results into a single DataFrame
    all_simulation_means = vcat(results...)

    return all_simulation_means
end