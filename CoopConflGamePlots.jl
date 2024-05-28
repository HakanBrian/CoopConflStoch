using Plots


##################
# game functions
##################

include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")


##################
# population construction
##################

my_parameter = simulation_parameters(0.6, 0.5, 0.45, 0.0, 500, 15, 50, 0.0, 0.05, 0.05, 0.05, 5)

my_population = population_construction(my_parameter)


##################
# simulation
##################

my_simulation = simulation(my_population)


##################
# plot
##################

plot(my_simulation.generation,
    [my_simulation.mean_action, my_simulation.mean_a, my_simulation.mean_p, my_simulation.mean_T, my_simulation.mean_payoff],
    title=string(my_parameter),
    titlefontsize=8,
    label=["action" "a" "p" "T" "payoff"])