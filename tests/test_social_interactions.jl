using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


######################
# Social Interactions ###########################################################################################################
######################

params = MainSimulation.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 10,
    group_size = 2,
    relatedness = 1.0,
)
population = MainSimulation.population_construction(params)

groups = MainSimulation.shuffle_and_group(
    population.groups,
    params.population_size,
    params.group_size,
    params.relatedness,
)
norm_pool = sum(@view population.norm[groups[1, :]]) / params.group_size
pun_pool = sum(@view population.ext_pun[groups[1, :]]) / params.group_size
action_sqrt = sqrt_llvm.(population.action)
action_sqrt_view = view(action_sqrt, groups[1, :])
action_sqrt_sum = sum(@view action_sqrt[groups[1, :]])

# Calculate behav eq
@time MainSimulation.behavioral_equilibrium!(
    groups[1, :],
    action_sqrt,
    action_sqrt_sum,
    norm_pool,
    pun_pool,
    population,
)

# Calculate best response
@code_warntype MainSimulation.best_response(
    1,
    groups[1, :],
    action_sqrt_view,
    action_sqrt_sum,
    norm_pool,
    pun_pool,
    population,
    0.1f0,
)

# Print actions
println(population.action)

# Calculate payoff
MainSimulation.total_payoff!(groups[1, :], norm_pool, pun_pool, population)
println(population.payoff)

# Calculate fitness
MainSimulation.fitness(population, groups[1, 1])

# Run social interactions
@time MainSimulation.social_interactions!(population)
println(population)
