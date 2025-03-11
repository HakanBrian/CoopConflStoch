using BenchmarkTools, Revise

push!(LOAD_PATH, abspath("src"));
using MainSimulation

# Apply changes
Revise.revise()


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
