using LinearAlgebra, Random, Distributions, StatsBase, DataFrames

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
    
    individuals_dict = Dict{Int64, agent}()
    for i in 1:parameters.N
        individual = agent(i, parameters.action0, parameters.a0, parameters.p0, parameters.T0, 0, 0, 0)
        individuals_dict[i] = individual
    end

    return population(parameters, individuals_dict, [], 0)
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

function payoff!(pairing::pair)
    benefit1 = √pairing.individual1.action
    benefit2 = √pairing.individual2.action

    cost1 = pairing.individual1.action^2
    cost2 = pairing.individual2.action^2

    norm_pool = mean([pairing.individual1.a, pairing.individual2.a])
    punishment_pool = mean([pairing.individual1.p, pairing.individual2.p])

    punishment1 = punishment_pool * (pairing.individual1.action - norm_pool)^2
    punishment2 = punishment_pool * (pairing.individual2.action - norm_pool)^2

    payoff1 = benefit2 - cost1 - punishment1
    payoff2 = benefit1 - cost2 - punishment2

    pairing.individual1.payoff = payoff1
    pairing.individual2.payoff = payoff2

    pairing.individual1.run_avg_payoff = (pairing.individual1.payoff + pairing.individual1.interactions * pairing.individual1.run_avg_payoff) / (pairing.individual1.interactions + 1)
    pairing.individual2.run_avg_payoff = (pairing.individual2.payoff + pairing.individual2.interactions * pairing.individual2.run_avg_payoff) / (pairing.individual2.interactions + 1)

    pairing.individual1.interactions += 1
    pairing.individual2.interactions += 1
    
    return pairing
end

function social_interactions!(pop::population)
    individuals_value = collect(value(pop.individuals))
    individuals_shuffle = shuffle(individuals_value)

    if pop.parameters.N % 2 != 0
        push!(individuals_shuffle, individuals_value[rand(1:pop.parameters.N)])
    end

    for i in 1:2:(length(individuals_shuffle)-1)
        push!(pop.pairings, pair(individuals_shuffle[i], individuals_shuffle[i+1]))
    end

    for i in 1:length(pop.pairings)
        payoff!(pop.pairings[i])
        pop.individuals[pop.pairings[i].individual1.id] = pop.pairings.individual1
        pop.individuals[pop.pairings[i].individual2.id] = pop.pairings.individual2
    end
end

##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents?
    # only need one parent
    # number of individuals in population remains the same

function reproduce!(pop::population)
    
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