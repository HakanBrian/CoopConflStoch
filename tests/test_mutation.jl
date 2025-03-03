using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


#########
# Mutate ########################################################################################################################
#########

params = MainSimulation.SimulationParameter()  # uses all default values
population = MainSimulation.population_construction(params);

# Create test mutate function
MainSimulation.mutate!(
    population,
    Simulations.truncation_bounds(population.parameters.mutation_variance, 0.99),
)
println(population)
