using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


##########################
# Population Construction #######################################################################################################
##########################

params = MainSimulation.SimulationParameter()  # uses all default values
population = MainSimulation.population_construction(params);
