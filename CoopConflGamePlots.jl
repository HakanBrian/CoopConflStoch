include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

using Plots

##################
# population_construction
##################

my_parameter = simulation_parameters(500, 20, 0.1, 0.05, 0.4, 0.55, 0.3, 0.0, 5)

my_population = population_construction(my_parameter)


#######################
# Simulation
#######################

my_simulation = simulation(my_population)

my_simulation.generation

plot(my_simulation.generation, my_simulation.mean_action)