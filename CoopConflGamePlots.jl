using Plots


##################
# game functions
##################

include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")


##################
# population construction
##################

my_parameter = simulation_parameters(0.25, 0.5, 0.2, 0.0, 500, 15, 10, 0.1, 0.0, 0.01, 5)

my_population = population_construction(my_parameter)


##################
# simulation
##################

my_simulation = simulation(my_population)


##################
# plot
##################

plot(my_simulation.generation, 
    [my_simulation.mean_action, my_simulation.mean_a, my_simulation.mean_p, my_simulation.mean_T], 
    title=string(my_parameter), 
    titlefontsize=10, 
    label=["action" "a" "p" "T"])