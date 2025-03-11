using BenchmarkTools, Revise

push!(LOAD_PATH, abspath("src"));
using MainSimulation

# Apply changes
Revise.revise()


##########################
# Population Construction #######################################################################################################
##########################

params = MainSimulation.SimulationParameter()  # uses all default values
population = MainSimulation.population_construction(params);
