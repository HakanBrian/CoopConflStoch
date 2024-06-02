using Random, Distributions, StatsBase, DataFrames, DifferentialEquations, ForwardDiff, StaticArrays


####################################
# Game Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation Funcs #
###############################

    # create a blank starting population
    # format output

function population_construction(parameters::simulation_parameters)
    individuals_dict = Dict{Int64, individual}()
    old_individuals_dict = Dict{Int64, individual}()
    use_distribution = parameters.trait_var != 0

    if use_distribution
        dist_values = Dict{Symbol, Any}()
        for name in fieldnames(simulation_parameters)[1:4]  # represents game params
            dist_values[name] = truncated(Normal(getfield(parameters, name), parameters.trait_var), 0, 1)
        end
    end

    for i in 1:parameters.N
        if use_distribution
            action0 = rand(dist_values[:action0])
            a0 = rand(dist_values[:a0])
            p0 = rand(dist_values[:p0])
            T0 = rand(dist_values[:T0])
        else
            action0 = parameters.action0
            a0 = parameters.a0
            p0 = parameters.p0
            T0 = parameters.T0
        end
        indiv = individual(action0, a0, p0, T0, 0.0, 0)
        individuals_dict[i] = indiv
        old_individuals_dict[i] = copy(indiv)
    end

    return population(parameters, individuals_dict, old_individuals_dict)
end


function output!(t::Int64, pop::population, outputs::DataFrame)
    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * pop.parameters.N + 1
    end

    # Iterate over all individuals in the population
    for i in 1:pop.parameters.N
        # Calculate the row for the current individual
        output_row = output_row_base + i - 1
        
        # Update the DataFrame with the individual's data
        outputs.generation[output_row] = t
        outputs.individual[output_row] = i
        outputs.action[output_row] = pop.individuals[i].action
        outputs.a[output_row] = pop.individuals[i].a
        outputs.p[output_row] = pop.individuals[i].p
        outputs.T[output_row] = pop.individuals[i].T
        outputs.payoff[output_row] = pop.individuals[i].payoff
    end
    nothing
end


##################
# Pairwise fitness
##################

    # pair individuals with the possibiliy of pairing more than once
    # everyone has the same chance of picking a partner / getting picked
    # at the end of the day everyone is picked roughly an equal number of times
    # aka random pairing without replacment

    # calculate payoff, and keep a running average of payoff for each individual
    # after each session of interaction the running average becomes the individual's payoff

@inline function benefit(action1::Any, action2::Float64, v::Float64)
    sqrt_action1 = √action1
    sqrt_action2 = √action2
    sqrt_sum = √(action1 + action2)
    return (1 - v) * (sqrt_action1 + sqrt_action2) + v * sqrt_sum
end

@inline function cost(action::Any)
    return action^2
end

@inline function norm_pool(a1::Float64, a2::Float64)
    return 0.5 * (a1 + a2)
end

@inline function punishment_pool(p1::Float64, p2::Float64)
    return 0.5 * (p1 + p2)
end

@inline function external_punishment(action::Any, a1::Float64, a2::Float64, p1::Float64, p2::Float64)
    norm = norm_pool(a1, a2)
    punishment = punishment_pool(p1, p2)
    return punishment * (action - norm)^2
end

@inline function internal_punishment(action::Any, a1::Float64, a2::Float64, T::Float64)
    norm = norm_pool(a1, a2)
    return T * (action - norm)^2
end

@inline function payoff(action1::Any, action2::Float64, a1::Float64, a2::Float64, p1::Float64, p2::Float64, v::Float64)
    return benefit(action1, action2, v) - cost(action1) - external_punishment(action1, a1, a2, p1, p2)
end

@inline function objective(action1::Any, action2::Float64, a1::Float64, a2::Float64, p1::Float64, p2::Float64, T::Float64, v::Float64)
    return payoff(action1, action2, a1, a2, p1, p2, v) - internal_punishment(action1, a1, a2, T)
end

@inline function objective_derivative(action1::Any, action2::Float64, a1::Float64, a2::Float64, p1::Float64, p2::Float64, T::Float64, v::Float64)
    return ForwardDiff.derivative(action1 -> objective(action1, action2, a1, a2, p1, p2, T, v), action1)
end

function total_payoff!(pair::Tuple{individual, individual}, parameters::simulation_parameters)
    payoff1 = payoff(pair[1].action, pair[2].action, pair[1].a, pair[2].a, pair[1].p, pair[2].p, parameters.v)
    payoff2 = payoff(pair[2].action, pair[1].action, pair[2].a, pair[1].a, pair[2].p, pair[1].p, parameters.v)

    pair[1].payoff = (payoff1 + pair[1].interactions * pair[1].payoff) / (pair[1].interactions + 1)
    pair[2].payoff = (payoff2 + pair[2].interactions * pair[2].payoff) / (pair[2].interactions + 1)

    pair[1].interactions += 1
    pair[2].interactions += 1

    nothing
end

function behav_ODESystem_static(u, p, t)
    dx = objective_derivative(u[1], u[2], p[1], p[2], p[3], p[4], p[5], p[7])
    dy = objective_derivative(u[2], u[1], p[2], p[1], p[4], p[3], p[6], p[7])

    return SA[dx, dy]
end

function behav_eq!(pair::Tuple{individual, individual}, parameters::simulation_parameters)
    u0 = SA[pair[1].action; pair[2].action]
    tspan = (0, parameters.tmax)
    p = SA[pair[1].a; pair[2].a; pair[1].p; pair[2].p; pair[1].T; pair[2].T; parameters.v]

    prob = ODEProblem(behav_ODESystem_static, u0, tspan, p)
    sol = solve(prob, Tsit5(), save_everystep = false)

    pair[1].action = sol[end][1]
    pair[2].action = sol[end][2]

    nothing
end

function social_interactions!(pop::population)
    individuals_key = collect(keys(copy(pop.individuals)))
    individuals_shuffle = shuffle(individuals_key)

    if pop.parameters.N % 2 != 0
        push!(individuals_shuffle, individuals_key[rand(1:pop.parameters.N)])
    end

    for i in 1:2:length(individuals_shuffle)-1
        behav_eq!((pop.individuals[individuals_shuffle[i]], pop.individuals[individuals_shuffle[i+1]]), pop.parameters)
        total_payoff!((pop.individuals[individuals_shuffle[i]], pop.individuals[individuals_shuffle[i+1]]), pop.parameters)
    end

    nothing
end


##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents
    # only need one parent
    # number of individuals in population remains the same

function reproduce!(pop::population)
    payoffs = [(individual.payoff) for individual in values(pop.individuals)]
    key = collect(keys(copy(pop.individuals)))
    genotype_array = sample(key, ProbabilityWeights(payoffs), pop.parameters.N, replace=true, ordered=false)

    copy!(pop.old_individuals, pop.individuals)
    for (res_i, offspring_i) in zip(key, genotype_array)
        pop.individuals[res_i] = pop.old_individuals[offspring_i]
    end

    nothing
end


##################
#  Mutation Function 
##################

    # offspring have slightly different trait values from their parents
    # use an independent draw function for each of the traits that could mutate

function mutate!(pop::population)
    for key in keys(pop.individuals)
        if rand() <= pop.parameters.u && pop.parameters.mut_var != 0
            a_dist = truncated(Normal(0, pop.parameters.mut_var), -pop.individuals[key].a, Inf)
            pop.individuals[key].a += rand(a_dist)
        end
        if rand() <= pop.parameters.u && pop.parameters.mut_var != 0
            p_dist = truncated(Normal(0, pop.parameters.mut_var), -pop.individuals[key].p, Inf)
            pop.individuals[key].p += rand(p_dist)
        end
        if rand() <= pop.parameters.u && pop.parameters.mut_var != 0
            T_dist = truncated(Normal(0, pop.parameters.mut_var), -pop.individuals[key].T, Inf)
            pop.individuals[key].T += rand(T_dist)
        end
    end

    nothing
end


#######################
# Simulation Function #
#######################

function simulation(pop::population)

    ############
    # Sim init #
    ############

    output_length = floor(Int64, pop.parameters.gmax/pop.parameters.output_save_tick) * pop.parameters.N
    outputs = DataFrame(generation = zeros(Int64, output_length),
                        individual = zeros(Int64, output_length),
                        action = zeros(Float64, output_length),
                        a = zeros(Float64, output_length),
                        p = zeros(Float64, output_length),
                        T = zeros(Float64, output_length),
                        payoff = zeros(Float64, output_length))

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
            output!(t, copy(pop), outputs)
        end

    end

return outputs
end