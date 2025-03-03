using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


############
# Reproduce #####################################################################################################################
############

# IMPORTANT: To use this test payoffs need to be copied into the next generation !!!

# Create sample population
param = MainSimulation.SimulationParameter(
    action0 = 0.5f0,
    norm0 = 0.5f0,
    ext_pun0 = 0.0f0,
    generations = 10,
    population_size = 1000,
    mutation_rate = 0.0,
)
population = MainSimulation.population_construction(param)
population.payoff[1:4] .= [1.0f0, 2.0f0, 3.0f0, 4.0f0]

# Bootstrap to increase sample size
original_size = 4
new_key = original_size + 1
for i in 1:original_size
    for j in 1:249
        population.payoff[new_key] = copy(population.payoff[i])
        new_key += 1
    end
end

# Ensure 250 copies of each parent
println(
    "Initial population with payoff 4: ",
    count(payoff -> payoff == 4.0f0, population.payoff),
)

# Complete a round of reproduction
MainSimulation.reproduce!(population)

# Offspring should have parent 4 as their parent ~40% of the time (only if there is no scaling)
println(
    "New population with payoff 4: ",
    count(payoff -> payoff == 4.0f0, population.payoff),
)
