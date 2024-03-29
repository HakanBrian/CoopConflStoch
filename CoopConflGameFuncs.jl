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
        individual = agent(i, 0, 0, 0, 0, 0, 0, 0)
        individuals_dict[i] = individual
    end
    
    return population(parameters, individuals_dict, [], 0)
end

function output(t::Int64, pop::population, outputs::DataFrame)

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

function benefit(pairing::pair)

end

function cost(pairing::pair)

end

function payoff(pairing::pair)

end

function social_interactions(pop::population)

end

##################
# Reproduction function
##################

    # offspring inherit the payoff or traits of the parents?
    # only need one parent
    # number of individuals in population remains the same

function reproduce(pop::population)
    
end

##################
#  Mutation Function 
##################

    # mutate?

function mutate(pop::population)

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

        # update population struct 
        update_population(pop)

        # execute social interactions and calculate payoffs
        social_interactions(pop)

        # reproduction function to produce and save t+1 population array
        reproduce(pop)

        # mutation function  iterates over population and mutates at chance probability μ
        if pop.parameters.μ > 0
            mutate(pop)
        end

        # per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output(t, copy(pop), outputs)
        end

    end
return outputs
end