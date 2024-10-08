####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")

@everywhere using StatsBase, Random, Distributions, DataFrames, StaticArrays, ForwardDiff, DiffEqGPU, DifferentialEquations, CUDA


###############################
# Population Simulation Function
###############################

@everywhere function offspring!(pop::Population, offspring_index::Int64, parent_index::Int64)
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

@everywhere function truncation_bounds(variance::Float64, retain_proportion::Float64)
    # Calculate tail probability alpha
    alpha = 1 - retain_proportion

    # Calculate z-score corresponding to alpha/2
    z_alpha_over_2 = quantile(Normal(), 1 - alpha/2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * √variance
    upper_bound = z_alpha_over_2 * √variance

    return SA[lower_bound, upper_bound]
end

@everywhere function population_construction(parameters::SimulationParameters)
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

@everywhere function output!(outputs::DataFrame, t::Int64, pop::Population)
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

@everywhere function benefit(action_i::Real, actions_j::AbstractVector{<:Real}, synergy::Real)
    sqrt_action_i = sqrt(action_i)
    sum_sqrt_actions_j = mapreduce(sqrt, +, actions_j)
    sum_sqrt_actions = sqrt_action_i + sum_sqrt_actions_j
    sqrt_sum_actions = sqrt(action_i + sum(actions_j))
    return (1 - synergy) * sum_sqrt_actions + synergy * sqrt_sum_actions
end

@everywhere function cost(action_i::Real)
    return action_i^2
end

@everywhere function external_punishment(action_i::Real, norm_pool::Real, punishment_pool::Real)
    return punishment_pool * (action_i - norm_pool)^2
end

@everywhere function internal_punishment_ext(action_i::Real, norm_pool_mini::Real, T_ext::Real)
    return T_ext * (action_i - norm_pool_mini)^2
end

@everywhere function internal_punishment_self(action_i::Real, norm_i::Real, T_self::Real)
    return T_self * (action_i - norm_i)^2
end

@everywhere function payoff(action_i::Real, actions_j::AbstractVector{<:Real}, norm_pool::Real, punishment_pool::Real, synergy::Real)
    return benefit(action_i, actions_j, synergy) - cost(action_i) - external_punishment(action_i, norm_pool, punishment_pool)
end

@everywhere function objective(action_i::Real, actions_j::AbstractVector{<:Real}, norm_i::Real, norm_mini::Real, norm_pool::Real, punishment_pool::Real, T_ext::Real, T_self::Real, synergy::Real)
    return payoff(action_i, actions_j, norm_pool, punishment_pool, synergy) - internal_punishment_ext(action_i, norm_mini, T_ext) - internal_punishment_self(action_i, norm_i, T_self)
end

@everywhere function objective_derivative(action_i::T, actions_j::AbstractVector{T}, norm_i::T, norm_mini::T, norm_pool::T, punishment_pool::T, T_ext::T, T_self::T, synergy::T) where T
    return ForwardDiff.derivative(x -> objective(x, actions_j, norm_i, norm_mini, norm_pool, punishment_pool, T_ext, T_self, synergy), action_i)
end

@everywhere function total_payoff!(group_indices::Vector{Int64}, group_norm::Float32, group_pun::Float32, pop::Population)
    # Extract the action of the focal individual as a real number (not a view)
    action_i = pop.action[group_indices[1]]

    # Collect actions from the other individuals in the group
    actions_j = @view pop.action[group_indices[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, group_norm, group_pun, pop.parameters.synergy)

    # Update the individual's payoff and interactions
    idx = group_indices[1]
    pop.payoff[idx] = (payoff_foc + pop.interactions[idx] * pop.payoff[idx]) / (pop.interactions[idx] + 1)
    pop.interactions[idx] += 1

    nothing
end

@everywhere function fitness(pop::Population, idx::Int64)
    return pop.payoff[idx] - pop.ext_pun[idx]
end

@everywhere function fitness(pop::Population, idx::Int64, fitness_scaling_factor_a::Float64, fitness_scaling_factor_b::Float64)
    base_fitness = fitness(pop, idx)
    return fitness_scaling_factor_a * exp(base_fitness * fitness_scaling_factor_b)
end


##################
# Behavioral Equilibrium Function
##################

@everywhere function remove_element(actions::SVector{N, T}, idx::Int64) where {N, T}
    return SVector{N-1}(ntuple(i -> i < idx ? actions[i] : actions[i + 1], N - 1))
end

@everywhere function behav_ODE_static(u::SVector{N, T}, p::SVector, t) where {N, T}
    du = ntuple(i -> objective_derivative(u[i], remove_element(u, i), p[1], p[2], p[3], p[4], p[5 + i], p[5 + i + N], p[5]), N)
    return SVector{N}(du)
end

@everywhere function behav_eq(action0s::Matrix{Float32}, int_pun_ext::Matrix{Float32}, int_pun_self::Matrix{Float32}, group_norm_means::Matrix{Float32}, group_pun_means::Vector{Float32}, parameters::SimulationParameters)
    synergy = parameters.synergy
    tmax = parameters.tmax
    num_groups = parameters.population_size
    group_size = parameters.group_size

    parameter_size = 5 + 2 * group_size

    # Create initial conditions for the first group
    u0 = SVector{group_size, Float32}(action0s[:, 1]...)  # Unpack the array into an SVector
    tspan = Float32[0.0f0, tmax]
    p = SVector{parameter_size, Float32}(
        group_norm_means[1, 1],      # Focal individual norm
        group_norm_means[2, 1],      # Other individuals' mean norm
        group_norm_means[3, 1],      # Group mean norm
        group_pun_means[1],          # Group punishment mean
        synergy,                     # Synergy parameter
        int_pun_ext[:, 1]...,        # External punishment vector unpacked
        int_pun_self[:, 1]...        # Internal punishment vector unpacked
    )

    # Create an initial ODE problem as a template
    prob = ODEProblem(behav_ODE_static, u0, tspan, p)

    # Function to remake the problem for each group
    prob_func = (prob, i, repeat) -> remake(prob,
                                            u0 = SVector{group_size, Float32}(action0s[:, i]...),
                                            p = SVector{parameter_size, Float32}(
                                                group_norm_means[1, i],
                                                group_norm_means[2, i],
                                                group_norm_means[3, i],
                                                group_pun_means[i],
                                                synergy,
                                                int_pun_ext[:, i]...,
                                                int_pun_self[:, i]...))

    # Create an ensemble problem
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    # Solve the ensemble problem using a GPU kernel
    sim = solve(ensemble_prob, EnsembleGPUKernel(CUDA.CUDABackend()), trajectories=num_groups, save_start=false, save_on=false)

    # Extract final action values for all groups
    final_actions = [sol[end] for sol in sim]

    return final_actions
end


##################
# Social Interactions Function
##################

@everywhere function probabilistic_round(x::Float64)::Int64
    lower = floor(Int64, x)
    upper = ceil(Int64, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

@everywhere function shuffle_and_group(population_size::Int64, group_size::Int64, relatedness::Float64)
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

@everywhere function collect_initial_conditions_and_parameters(groups::Matrix{Int64}, pop::Population)
    num_groups = pop.parameters.population_size
    group_size = pop.parameters.group_size

    # Preallocate arrays for the output
    action0s = zeros(Float32, group_size, num_groups)
    int_pun_ext = zeros(Float32, group_size, num_groups)
    int_pun_self = zeros(Float32, group_size, num_groups)

    # Array for group norms: (focal_norm, others_mean_norm, group_mean_norm)
    group_norm_means = zeros(Float32, 3, num_groups)

    # Array for group punishment means
    group_pun_means = zeros(Float32, num_groups)

    # Preallocate temporary storage for group norms and punishment pools outside the loop
    group_norms = zeros(Float32, group_size)
    pun_pools = zeros(Float32, group_size)

    # Loop over each group and collect data and compute norms and punishments simultaneously
    for i in 1:num_groups
        # Create a view into the i-th row of the groups matrix
        group = @view groups[i, :]

        # Loop over each group member and collect data
        for j in 1:group_size
            member_index = group[j]
            action0s[j, i] = pop.action[member_index]  # Collect actions
            group_norms[j] = pop.norm[member_index]  # Collect norms
            pun_pools[j] = pop.ext_pun[member_index]  # Collect punishment
            int_pun_ext[j, i] = pop.int_pun_ext[member_index]  # Collect external internal punishment
            int_pun_self[j, i] = pop.int_pun_self[member_index]  # Collect self internal punsihment
        end

        # Compute focal norm, others' mean norm, and group mean norm
        focal_norm = group_norms[1]
        others_mean_norm = mean(group_norms[2:end])
        group_mean_norm = mean(group_norms)
        group_norm_means[:, i] = [focal_norm, others_mean_norm, group_mean_norm]

        # Compute group mean punishment
        group_pun_means[i] = mean(pun_pools)
    end

    return action0s, int_pun_ext, int_pun_self, group_norm_means, group_pun_means
end

@everywhere function update_actions_and_payoffs!(final_actions::Vector{SVector{N, Float32}}, groups::Matrix{Int64}, group_norm_means::Matrix{Float32}, group_pun_pools::Vector{Float32}, pop::Population) where N
    action_variance = 0.0  # Placeholder for now
    use_distribution = action_variance != 0

    if use_distribution
        lower_bound, upper_bound = truncation_bounds(action_variance, 0.99)
    end

    population_size = pop.parameters.population_size

    for j in 1:population_size
        actions = final_actions[j]
        group_indices = Vector(@view groups[j, :])  # Get group members' indices

        # Update the action for each individual in the group
        for (i, idx) in enumerate(group_indices)
            final_action = actions[i]

            if use_distribution
                final_actions_dist = truncated(Normal(0, action_variance), lower=max(lower_bound, -final_action), upper=upper_bound)
                pop.action[idx] = final_action + rand(final_actions_dist)
            else
                pop.action[idx] = final_action
            end
        end

        # Update payoffs
        total_payoff!(group_indices, group_norm_means[3, j], group_pun_pools[j], pop)
    end

    nothing
end

@everywhere function social_interactions!(pop::Population)
    # Shuffle and pair individuals
    groups = shuffle_and_group(pop.parameters.population_size, pop.parameters.group_size, pop.parameters.relatedness)

    # Calculate final actions for all pairs
    action0s, int_pun_ext, int_pun_self, group_norm_means, group_pun_means = collect_initial_conditions_and_parameters(groups, pop)
    final_actions = behav_eq(action0s, int_pun_ext, int_pun_self, group_norm_means, group_pun_means, pop.parameters)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, groups, group_norm_means, group_pun_means, pop)

    nothing
end


##################
# Reproduction Function
##################

@everywhere function reproduce!(pop::Population)
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
@everywhere function reproduce!(pop::Population)
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

@everywhere function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
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
@everywhere function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
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

@everywhere function simulation(pop::Population)

    ############
    # Sim init #
    ############

    output_length = floor(Int64, pop.parameters.gmax / pop.parameters.output_save_tick) * pop.parameters.population_size
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