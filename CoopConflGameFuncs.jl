using Optimization, OptimizationOptimJL, LinearAlgebra, Random, Distributions, StatsBase, DataFrames

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
    # constructs a population array when supplied with parameters
    
    action0_dist = Truncated(Normal(parameters.action0, parameters.var), 0, 1)
    a0_dist = Truncated(Normal(parameters.a0, parameters.var), 0, 1)
    p0_dist = Truncated(Normal(parameters.p0, parameters.var), 0, 1)
    T0_dist = Truncated(Normal(parameters.T0, parameters.var), 0, 1)

    individuals_dict = Dict{Int64, individual}()
    for i in 1:parameters.N
        individuals_dict[i] = individual(rand(action0_dist), rand(a0_dist), rand(p0_dist), rand(T0_dist), 0, 0)
    end

    return population(parameters, individuals_dict, 0)
end

function output!(t::Int64, pop::population, outputs::DataFrame)

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

function benefit(action::Float64)
    return √action
end

function cost(action::Float64)
    return action^2
end

function norm_pool(a1::Float64, a2::Float64)
    return mean([a1, a2])
end

function punishment_pool(p1::Float64, p2::Float64)
    return mean([p1, p2])
end

function external_punishment(action::Float64, a1::Float64, a2::Float64, p1::Float64, p2::Float64)
    return punishment_pool(p1, p2) * (action - norm_pool(a1, a2))^2
end

function internal_punishment(action::Float64, a1::Float64, a2::Float64, T::Float64)
    return T * (action - norm_pool(a1, a2))^2
end

function payoff(individual1::individual, individual2::individual)
    return benefit(individual2.action) - cost(indiviual1.action) - external_punishment(individual1.action, individual1.a, individual2.a, individual1.p, individual2.p)
end

function objective(individual1::individual, individual2::individual)
    return payoff(individual1, individual2) - internal_punishment(individual1.action, individual1.a, individual2.a, individual1.T)
end

function total_payoff!(individual1::individual, individual2::individual)
    payoff1 = payoff(individual1, individual2)
    payoff2 = payoff(individual2, individual1)

    individual1.payoff = (payoff1 + individual1.interactions * individual1.payoff) / (individual1.interactions + 1)
    individual2.payoff = (payoff2 + individual2.interactions * individual2.payoff) / (individual2.interactions + 1)

    individual1.interactions += 1
    individual2.interactions += 1
end

function behav_eq!(individual1::individual, individual2::individual)
    action10 = individual1.action
    action20 = individual2.action
end

function social_interactions!(pop::population)
    individuals_key = collect(keys(copy(pop.individuals)))
    individuals_shuffle = shuffle(individuals_key)

    if pop.parameters.N % 2 != 0
        push!(individuals_shuffle, individuals_key[rand(1:pop.parameters.N)])
    end

    for i in 1:2:(length(individuals_shuffle)-1)
        behav_eq!(pop.individuals[individuals_shuffle[i]], pop.individuals[individuals_shuffle[i+1]])
        total_payoff!(pop.individuals[individuals_shuffle[i]], pop.individuals[individuals_shuffle[i+1]])
    end
end

##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents?
    # only need one parent
    # number of individuals in population remains the same

function reproduce!(pop::population)
    payoffs = [individual.payoff for individual in values(pop.individuals)]
    pop.mean_w = mean(payoffs)
    genotype_array = sample(1:pop.parameters.N, ProbabilityWeights(payoffs), pop.parameters.N, replace=true)
end

##################
#  Mutation Function 
##################

    # mutate?

function mutate!(pop::population)

end

#######################
# Simulation Function #
#######################

function simulation(pop::population)

    ############
    # Sim init #
    ############
    
    outputs = DataFrame()
    
    ############
    # Sim Loop #
    ############
      
    for t in 1:pop.parameters.tmax

        # execute social interactions and calculate payoffs
        social_interactions!(pop)

        # reproduction function to produce and save t+1 population array
        reproduce!(pop)

        # mutation function  iterates over population and mutates at chance probability μ
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