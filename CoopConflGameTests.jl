using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Population Construction
##################

params = SimulationParameters()  # uses all default values
population = population_construction(params);


##################
# BehavEq & Payoff & Fitness & Social Interactions
##################

params = SimulationParameters(action0=0.1f0, norm0=2.0f0, ext_pun0=0.1f0, int_pun_ext0=0.0f0, int_pun_self0=0.0f0, population_size=50, group_size=10, relatedness=1.0)
population = population_construction(params)

groups = shuffle_and_group(params.population_size, params.group_size, params.relatedness)

# Calculate behav eq
@time best_actions = behavioral_equilibrium(groups[1, :], population)

# Check payoff
total_payoff!(best_actions, groups[1, :], population)
println(population.payoff)

# Calculate fitness
fitness(population, groups[1, 1])

@time social_interactions!(population)
println(population)


##################
# Reproduce
##################

# IMPORTANT: To use this test payoffs need to be copied into the next generation !!!

# Create sample population
param = SimulationParameters(action0=0.5f0, norm0=0.5f0, ext_pun0=0.0f0, generations=10, population_size=1000, mutation_rate=0.0)
population = population_construction(param)
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
println("Initial population with payoff 4: ", count(payoff -> payoff == 4.0f0, population.payoff))

# Complete a round of reproduction
reproduce!(population)

# Offspring should have parent 4 as their parent ~40% of the time (only if there is no scaling)
println("New population with payoff 4: ", count(payoff -> payoff == 4.0f0, population.payoff))


##################
# Mutate
##################

mutate!(population, truncation_bounds(my_population.parameters.mutation_variance, 0.99))
println(population)


##################
# Profiling
##################

# compilation
@time simulation(population);
# pure runtime
@profview @time simulation(population);