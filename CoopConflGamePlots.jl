using Plots


##################
# game functions
##################

include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")


##################
# population construction
##################

my_parameter = simulation_parameters(0.6, 0.5, 0.45, 0.0, 10, 15, 5, 0.0, 0.05, 0.05, 0.05, 1)

my_population = population_construction(my_parameter)


##################
# simulation
##################

my_simulation = simulation(my_population)
println(my_simulation)

my_simulation_gdf = groupby(my_simulation, :generation)
my_simulation_mean = combine(my_simulation_gdf, :action => mean, :a => mean, :p => mean, :T => mean, :payoff => mean)


##################
# plot
##################

plot(my_simulation_mean.generation,
    [my_simulation_mean.action_mean, my_simulation_mean.a_mean, my_simulation_mean.p_mean, my_simulation_mean.T_mean, my_simulation_mean.payoff_mean],
    title=string(my_parameter),
    titlefontsize=8,
    label=["action" "a" "p" "T" "payoff"])