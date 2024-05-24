using Plots


##################
# game functions
##################

include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")


##################
# population construction
##################

my_parameter = simulation_parameters(0.4, 0.55, 0.3, 0.0, 500, 20, 0.1, 0.05, 0.05, 5)

my_population = population_construction(my_parameter)


##################
# simulation
##################

my_simulation = simulation(my_population)


##################
# plot
##################

plot(my_simulation.generation, [my_simulation.mean_action, my_simulation.mean_a, my_simulation.mean_p, my_simulation.mean_T], title="Evo of Traits", label=["action" "a" "p" "T"])