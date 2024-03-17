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

##################
# Pairwise fitness
##################

function social_interactions!(pop::population)

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
    
    output_length = floor(Int64, pop.parameters.tmax/pop.parameters.output_save_tick)
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