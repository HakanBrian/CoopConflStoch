using Random, Distributions, StatsBase, DataFrames, DifferentialEquations, ForwardDiff, StaticArrays, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Funcs #
###############################

    # Create an initial population
    # Format output

function population_construction(parameters::simulation_parameters)
    individuals_dict = Dict{Int64, individual}()
    old_individuals_dict = Dict{Int64, individual}()

    trait_var = parameters.trait_var
    use_distribution = trait_var != 0

    # Collect initial traits
    action0 = parameters.action0
    a0 = parameters.a0
    p0 = parameters.p0
    T0 = parameters.T0

    # Construct distributions
    if use_distribution
        action0_dist = truncated(Normal(action0, trait_var), lower=0)
        a0_dist = truncated(Normal(a0, trait_var), lower=0)
        p0_dist = truncated(Normal(p0, trait_var), lower=0)
        T0_dist = truncated(Normal(T0, trait_var), lower=0)
    end

    # Create individuals
    for i in 1:parameters.N
        if use_distribution
            action0 = rand(action0_dist)
            a0 = rand(a0_dist)
            p0 = rand(p0_dist)
            T0 = rand(T0_dist)
        end
        indiv = individual(action0, a0, p0, T0, 0.0, 0)
        individuals_dict[i] = indiv
        old_individuals_dict[i] = copy(indiv)
    end

    return population(parameters, individuals_dict, old_individuals_dict)
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

    return nothing
end


##################
# Fitness function
##################

    # Calculate payoff, and keep a running average of payoff for each individual
    # After each session of interaction the running average becomes the individual's payoff

function benefit(action1::Any, action2::Any, v::Any)
    sqrt_action1 = √max(action1, 0)
    sqrt_action2 = √max(action2, 0)
    sqrt_sum = √max((action1 + action2), 0)
    return (1 - v) * (sqrt_action1 + sqrt_action2) + v * sqrt_sum
end

function cost(action::Any)
    return action^2
end

function norm_pool(a1::Any, a2::Any)
    return 0.5 * (a1 + a2)
end

function punishment_pool(p1::Any, p2::Any)
    return 0.5 * (p1 + p2)
end

function external_punishment(action::Any, a1::Any, a2::Any, p1::Any, p2::Any)
    norm = norm_pool(a1, a2)
    punishment = punishment_pool(p1, p2)
    return punishment * (action - norm)^2
end

function internal_punishment(action::Any, a1::Any, a2::Any, T::Any)
    norm = norm_pool(a1, a2)
    return T * (action - norm)^2
end

function payoff(action1::Any, action2::Any, a1::Any, a2::Any, p1::Any, p2::Any, v::Any)
    return benefit(action1, action2, v) - cost(action1) - external_punishment(action1, a1, a2, p1, p2)
end

function objective(action1::Any, action2::Any, a1::Any, a2::Any, p1::Any, p2::Any, T::Any, v::Any)
    return payoff(action1, action2, a1, a2, p1, p2, v) - internal_punishment(action1, a1, a2, T)
end

function objective_derivative(action1::Any, action2::Any, a1::Any, a2::Any, p1::Any, p2::Any, T::Any, v::Any)
    return ForwardDiff.derivative(action1 -> objective(action1, action2, a1, a2, p1, p2, T, v), action1)
end

function total_payoff!(ind1::individual, ind2::individual, v::Float64)
    payoff1 = max(payoff(ind1.action, ind2.action, ind1.a, ind2.a, ind1.p, ind2.p, v), 0)
    payoff2 = max(payoff(ind2.action, ind1.action, ind2.a, ind1.a, ind2.p, ind1.p, v), 0)

    ind1.payoff = (payoff1 + ind1.interactions * ind1.payoff) / (ind1.interactions + 1)
    ind2.payoff = (payoff2 + ind2.interactions * ind2.payoff) / (ind2.interactions + 1)

    ind1.interactions += 1
    ind2.interactions += 1

    return nothing
end


##################
# Behavioral Equilibrium function
##################

function behav_ODESystem_static(u, p, t)
    dx = objective_derivative(u[1], u[2], p[1], p[2], p[3], p[4], p[5], p[7])
    dy = objective_derivative(u[2], u[1], p[2], p[1], p[4], p[3], p[6], p[7])

    return SA[dx, dy]
end

function behav_eq!(ind1::individual, ind2::individual, tmax::Int64, v::Float64)
    # Collect variables, timespan, and parameters
    u0 = SA[ind1.action; ind2.action]
    tspan = (0, tmax)
    p = SA[ind1.a; ind2.a; ind1.p; ind2.p; ind1.T; ind2.T; v]

    # Define and solve the problem
    prob = ODEProblem(behav_ODESystem_static, u0, tspan, p)
    sol = solve(prob, Tsit5(), save_everystep = false)

    # Update action values
    ind1.action, ind2.action = sol[end]

    return nothing
end

# Define the model
@mtkmodel BEHAV_ODE begin
    @parameters begin
        a1
        a2
        p1
        p2
        T1
        T2
        v
    end
    @variables begin
        action1(t)
        action2(t)
    end
    @equations begin
        D(action1) ~ objective_derivative(action1, action2, a1, a2, p1, p2, T1, v)
        D(action2) ~ objective_derivative(action2, action1, a2, a1, p2, p1, T2, v)
    end
end

# Build the model
@mtkbuild behav_ODE = BEHAV_ODE()

function behav_eq_MTK!(ind1::individual, ind2::individual, tmax::Int64, v::Float64)
    # Collect fields
    u0 = [behav_ODE.action1 => ind1.action, behav_ODE.action2 => ind2.action]
    tspan = (0, tmax)
    p = [behav_ODE.a1 => ind1.a, behav_ODE.a2 => ind2.a, behav_ODE.p1 => ind1.p, behav_ODE.p2 => ind2.p, behav_ODE.T1 => ind1.T, behav_ODE.T2 => ind2.T, behav_ODE.v => v]

    # Define and solve the problem
    prob = ODEProblem(behav_ODE, u0, tspan, p)
    sol = solve(prob, Tsit5(), save_everystep = false)

    # Update action values
    ind1.action, ind2.action = sol[end]

    return nothing
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

    # Local variables for frequently accessed property
    N = pop.parameters.N
    tmax = pop.parameters.tmax
    v = pop.parameters.v

    # If the number of individuals is odd, append a random individual to the shuffled list
    if N % 2 != 0
        push!(individuals_shuffle, individuals_key[rand(1:N)])
    end

    # Iterate over the pairs of individuals
    for i in 1:2:N-1
        ind1 = pop.individuals[individuals_shuffle[i]]
        ind2 = pop.individuals[individuals_shuffle[i+1]]
        behav_eq!(ind1, ind2, tmax, v)
        total_payoff!(ind1, ind2, v)
    end

    return nothing
end


##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents
    # number of individuals in population remains the same

function reproduce!(pop::population)
    individuals = values(pop.individuals)
    payoffs = map(individual -> individual.payoff, individuals)
    keys_list = collect(keys(pop.individuals))

    # Sample with the given weights
    sampled_keys = sample(keys_list, ProbabilityWeights(payoffs), pop.parameters.N, replace=true, ordered=false)

    # Temporarily store old individuals
    copy!(pop.old_individuals, pop.individuals)

    # Update population individuals based on sampled keys
    for (key, sampled_key) in zip(keys_list, sampled_keys)
        pop.individuals[key] = pop.old_individuals[sampled_key]
    end

    return nothing
end


##################
#  Mutation Function 
##################

    # offspring have slightly different trait values from their parents
    # use an independent draw function for each of the traits that could mutate

function mutate!(pop::population)
    u = pop.parameters.u
    mut_var = pop.parameters.mut_var

    # Only mutate if necessary
    if mut_var == 0
        return nothing
    end

    # Indpendent draw for each of the traits to mutate
    for key in keys(pop.individuals)
        ind = pop.individuals[key]

        if rand() <= u
            ind.a = rand(truncated(Normal(ind.a, mut_var), lower=0))
        end

        if rand() <= u
            ind.p = rand(truncated(Normal(ind.p, mut_var), lower=0))
        end

        if rand() <= u
            ind.T = rand(truncated(Normal(ind.p, mut_var), lower=0))
        end
    end

    return nothing
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

    ############
    # Sim Loop #
    ############

    for t in 1:pop.parameters.gmax
        # execute social interactions and calculate payoffs
        social_interactions!(pop)

        # reproduction function to produce new generation
        reproduce!(pop)

        # mutation function iterates over population and mutates at chance probability μ
        if pop.parameters.u > 0
            mutate!(pop)
        end

        # per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(outputs, t, copy(pop))
        end
    end

    return outputs
end