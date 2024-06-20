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

function population_construction(parameters::Simulation_Parameters)
    N = parameters.N
    actions = zeros(Float64, N)
    as = zeros(Float64, N)
    ps = zeros(Float64, N)
    Ts = zeros(Float64, N)
    payoffs = zeros(Float64, N)
    interactions = zeros(Int64, N)

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
            actions[i] = action0 + rand(action0_dist)
            as[i] = a0 + rand(a0_dist)
            ps[i] = p0 + rand(p0_dist)
            Ts[i] = T0 + rand(T0_dist)
        else
            actions[i] = action0
            as[i] = a0
            ps[i] = p0
            Ts[i] = T0 
        end
    end

    return Population(parameters, actions, as, ps, Ts, payoffs, interactions, 0.0, 0.0)
end

function output!(outputs::DataFrame, t::Int64, pop::Population)
    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * pop.parameters.N + 1
    end

    N = pop.parameters.N

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base + N - 1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = fill(t, N)
    outputs.individual[output_rows] = 1:N
    outputs.action[output_rows] = pop.actions[1:N]
    outputs.a[output_rows] = pop.as[1:N]
    outputs.p[output_rows] = pop.ps[1:N]
    outputs.T[output_rows] = pop.Ts[1:N]
    outputs.payoff[output_rows] = pop.payoffs[1:N]

    nothing
end


##################
# Fitness function
##################

    # Calculate payoff, and keep a running average of payoff for each individual
    # After each session of interaction the running average becomes the individual's payoff

function benefit(action1::Real, action2::Real, v::Real)
    sqrt_action1 = √max(action1, 0)
    sqrt_action2 = √max(action2, 0)
    sqrt_sum = √max((action1 + action2), 0)
    return (1 - v) * (sqrt_action1 + sqrt_action2) + v * sqrt_sum
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

function payoff(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, v::Real)
    return benefit(action1, action2, v) - cost(action1) - external_punishment(action1, norm_pool, punishment_pool)
end

function objective(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, T::Real, v::Real)
    return payoff(action1, action2, norm_pool, punishment_pool, v) - internal_punishment(action1, norm_pool, T)
end

function objective_derivative(action1::Real, action2::Real, norm_pool::Real, punishment_pool::Real, T::Real, v::Real)
    return ForwardDiff.derivative(x -> objective(x, action2, norm_pool, punishment_pool, T, v), action1)
end

function total_payoff!(pop::Population, idx1::Int, idx2::Int, norm_pool::Float64, punishment_pool::Float64, v::Float64)
    ind1_action = pop.actions[idx1]
    ind2_action = pop.actions[idx2]

    payoff1 = max(payoff(ind1_action, ind2_action, norm_pool, punishment_pool, v), 0)
    payoff2 = max(payoff(ind2_action, ind1_action, norm_pool, punishment_pool, v), 0)

    # Update payoffs for individual at idx1
    pop.payoffs[idx1] = (payoff1 + pop.interactions[idx1] * pop.payoffs[idx1]) / (pop.interactions[idx1] + 1)
    pop.interactions[idx1] += 1

    # Update payoffs for individual at idx2
    pop.payoffs[idx2] = (payoff2 + pop.interactions[idx2] * pop.payoffs[idx2]) / (pop.interactions[idx2] + 1)
    pop.interactions[idx2] += 1

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

function behav_eq!(pairs::Vector{Tuple{Individual, Individual}}, norm_pool::Float64, punishment_pool::Float64, tmax::Float32, v::Float64)
    # Extract initial conditions and parameters
    u0s = [SA_F32[ind1.action, ind2.action] for (ind1, ind2) in pairs]
    tspan = (0.0f0, tmax)
    ps = [SA_F32[norm_pool, punishment_pool, ind1.T, ind2.T, v] for (ind1, ind2) in pairs]

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

function shuffle_and_pair(individuals_indices::Vector{Int64})
    shuffle!(individuals_indices)
    num_pairs = length(individuals_indices) ÷ 2
    pairs = [(individuals_indices[2i-1], individuals_indices[2i]) for i in 1:num_pairs]

    if isodd(length(individuals_indices))
        # Pair the last individual with a random individual
        push!(pairs, (individuals_indices[end], rand(individuals_indices[1:end-1])))
        num_pairs += 1
    end

    return pairs, num_pairs
end

function collect_initial_conditions_and_parameters(pairs::Vector{Tuple{Int64, Int64}}, num_pairs::Int64, pop::Population)
    u0s = Vector{SArray{Tuple{2}, Float32}}(undef, num_pairs)
    ps = Vector{SArray{Tuple{5}, Float32}}(undef, num_pairs)

    for (i, (idx1, idx2)) in enumerate(pairs)
        u0s[i] = SA_F32[pop.actions[idx1]; pop.actions[idx2]]
        ps[i] = SA_F32[pop.norm_pool, pop.punishment_pool, pop.Ts[idx1], pop.Ts[idx2], pop.parameters.v]
    end

    return u0s, ps
end

function update_actions_and_payoffs!(final_actions::Vector{SVector{2, Float32}}, pairs::Vector{Tuple{Int64, Int64}}, pop::Population)
    for (i, (idx1, idx2)) in enumerate(pairs)
        pop.actions[idx1], pop.actions[idx2] = final_actions[i]
        total_payoff!(pop, idx1, idx2, pop.norm_pool, pop.punishment_pool, pop.parameters.v)
    end

    nothing
end

function social_interactions!(pop::Population)
    individuals_indices = collect(1:pop.parameters.N)

    # Update norm and punishment pools
    pop.norm_pool = mean(pop.as)
    pop.punishment_pool = mean(pop.ps)

    # Shuffle and pair individuals
    pairs, num_pairs = shuffle_and_pair(individuals_indices)

    # Collect initial conditions and parameters for all pairs
    u0s, ps = collect_initial_conditions_and_parameters(pairs, num_pairs, pop)

    # Calculate final actions for all pairs
    final_actions = behav_eq(u0s, ps, pop.parameters.tmax, num_pairs)

    # Update actions and payoffs for all pairs based on final actions
    update_actions_and_payoffs!(final_actions, pairs, pop)

    nothing
end


##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents
    # number of individuals in population remains the same

function reproduce!(pop::Population)
    # Uncomment below if testing selection
    # payoffs = map(payoff -> 1 + 0.5 * payoff, pop.payoffs)

    payoffs = pop.payoffs

    # Sample with the given weights
    sampled_idxs = sample(1:pop.parameters.N, ProbabilityWeights(payoffs), pop.parameters.N, replace=true)

    # Sort the sampled indices
    sort!(sampled_idxs)

    # Update population individuals based on sampled keys
    for (i, sampled_idx) in enumerate(sampled_idxs)
        parent = get_individual(pop, sampled_idx)
        set_individual!(pop, i, parent)
    end

    nothing
end


##################
#  Mutation Function 
##################

    # offspring have slightly different trait values from their parents
    # use an independent draw function for each of the traits that could mutate

function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2}, Float64})
    mut_var = pop.parameters.mut_var

    # Only mutate if necessary
    if mut_var == 0
        return nothing
    end

    u = pop.parameters.u
    lower_bound, upper_bound = truncate_bounds

    # Mutate each trait independently for all individuals
    for i in 1:pop.parameters.N
        if rand() <= u
            # Mutate 'a' trait
            a_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -pop.as[i]), upper=upper_bound)
            pop.as[i] += rand(a_dist)
        end

        if rand() <= u
            # Mutate 'p' trait
            p_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -pop.ps[i]), upper=upper_bound)
            pop.ps[i] += rand(p_dist)
        end

        # Uncomment below if 'T' trait mutation is required
        # if rand() <= u
        #     T_dist = truncated(Normal(0, mut_var), lower=max(lower_bound, -pop.Ts[i]), upper=upper_bound)
        #     pop.Ts[i] += rand(T_dist)
        # end
    end

    nothing
end


#######################
# Simulation Function #
#######################

function simulation(pop::Population)

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