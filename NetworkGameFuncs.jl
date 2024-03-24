using LinearAlgebra, Random, Distributions, ArgParse, StatsBase, DataFrames

####################################
# Network Game Functions
####################################

include("NetworkGameStructs.jl")

###############################
# Population Simulation Funcs #
###############################

function update_population!(pop::population)

end

function population_construction(parameters::simulation_parameters)
    ## constructs a population array when supplied with parameters
    return population(parameters, shuffle(1:parameters.N), zeros(Float64, parameters.N),zeros(Float64, parameters.N),zeros(Float64, parameters.N),zeros(Float64, parameters.N),zeros(Float64, parameters.N),0)
end

##################
# Pairwise fitness
##################

function social_interactions!(pop::population)
    # match individuals randomly in pairs
    if pop.parameters.N % 2 == 0
        pop.shuffled_indices = shuffle(1:pop.parameters.N)
    else
        # if population size is odd, last individual plays themself
        shfld = shuffle(1:pop.parameters.N)
        pop.shuffled_indices = [shfld; shfld[end]]
    end

    return pop.shuffled_indices
end

##################
# Reproduction function
##################

function reproduce!(pop::population)
    pop.mean_w = mean(pop.payoffs)
    
end

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
        update_population!(pop)
    
        # execute social interactions and calculate payoffs
        social_interactions!(pop)
    
        # reproduction function to produce and save t+1 population array
        reproduce!(pop)
    
        # per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(t, copy(pop), outputs)
        end
    
    end
return outputs
end