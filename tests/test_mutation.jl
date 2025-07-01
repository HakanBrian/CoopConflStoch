using BenchmarkTools, Revise, Distributions

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

# Truncated distribtion has mean shift
mutation_dist = Normal(0, 0.005)
mean(mutation_dist)

truncate_bounds = MainSimulation.Populations.truncation_bounds(0.005, 0.99)

test_dist = truncated(mutation_dist, 0, truncate_bounds[2])
mean(test_dist)
std(test_dist)
