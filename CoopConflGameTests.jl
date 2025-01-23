using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")
include("CoopConflGamePlots.jl")
include("SLURM/CoopConflGameSLURMHelper.jl")


##################
# Population Construction
##################

params = SimulationParameters()  # uses all default values
population = population_construction(params);


##################
# BehavEq & Payoff & Fitness & Social Interactions
##################

params = SimulationParameters(action0=0.1f0, norm0=2.0f0, ext_pun0=0.1f0, int_pun_ext0=0.0f0, int_pun_self0=0.0f0, population_size=10, group_size=2, relatedness=1.0)
population = population_construction(params)

groups = shuffle_and_group(population.groups ,params.population_size, params.group_size, params.relatedness)
norm_pool = sum(@view population.norm[groups[1, :]]) / params.group_size
pun_pool = sum(@view population.ext_pun[groups[1, :]]) / params.group_size
action_sqrt = sqrt_llvm.(population.action)
action_sqrt_sum = sum(@view action_sqrt[groups[1, :]])

# Calculate behav eq
@time behavioral_equilibrium!(groups[1, :], action_sqrt, action_sqrt_sum, norm_pool, pun_pool, population)
println(population.action)

# Calculate payoff
total_payoff!(groups[1, :], norm_pool, pun_pool, population)
println(population.payoff)

# Calculate fitness
fitness(population, groups[1, 1])

# Run social interactions
@time social_interactions!(population)
println(population)


##################
# Behavior of BehavEq
##################

base_params = SimulationParameters(action0=0.0f0,
                                    norm0=0.0f0,
                                    ext_pun0=0.0f0,
                                    int_pun_ext0=0.0f0,
                                    int_pun_self0=0.0f0,
                                    generations=100000,
                                    population_size=10,
                                    group_size=10,
                                    relatedness=0.0,
                                    ext_pun_mutation_enabled=true,
                                    int_pun_ext_mutation_enabled=true,
                                    int_pun_self_mutation_enabled=true,
                                    output_save_tick=10)

action_values = collect(range(0.0f0, 10.0f0, step=0.01f0));
norm_values = collect(range(0.0f0, 10.0f0, step=0.01f0));
ext_pun_values = collect(range(0.0f0, 10.0f0, step=0.01f0));
int_pun_ext_values = collect(range(0.0f0, 10.0f0, step=0.01f0));
int_pun_self_values = collect(range(0.0f0, 10.0f0, step=0.01f0));

parameter_sweep_action = [
    update_params(base_params, action0=action_value)
    for action_value in action_values
]
parameter_sweep_norm = [
    update_params(base_params, norm0=norm_value)
    for norm_value in norm_values
]
parameter_sweep_ext_pun = [
    update_params(base_params, ext_pun0=ext_pun_value)
    for ext_pun_value in ext_pun_values
]
parameter_sweep_int_pun_ext = [
    update_params(base_params, int_pun_ext0=int_pun_ext_value)
    for int_pun_ext_value in int_pun_ext_values
]
parameter_sweep_int_pun_self = [
    update_params(base_params, int_pun_self0=int_pun_self_value)
    for int_pun_self_value in int_pun_self_values
]

function test_behav_eq(param_sweep::Vector{SimulationParameters})
    actions = Vector{Float32}(undef, length(param_sweep))

    for (i, param) in enumerate(param_sweep)
        population = population_construction(param)
        groups = shuffle_and_group(population.groups, param.population_size, param.group_size, param.relatedness)
        norm_pool, pun_pool = collect_group(groups[1, :], population)
        action_sqrt = sqrt_llvm.(population.action)
        buffer = Vector{Float32}(undef, param.group_size - 1)
        behavioral_equilibrium!(groups[1, :], buffer, action_sqrt, norm_pool, pun_pool, population)
        actions[i] = mean(population.action)
    end

    p = Plots.plot(action_values,
                   actions,    
                   legend=true,
                   label="Action",
                   xlabel="initial",
                   ylabel="Action",
                   title="Behavioral Equilibrium",
                   size=(800, 600)
    )

    # Display the plot
    display("image/png", p)
end

test_behav_eq(parameter_sweep_action)
test_behav_eq(parameter_sweep_norm)
test_behav_eq(parameter_sweep_ext_pun)
test_behav_eq(parameter_sweep_int_pun_ext)
test_behav_eq(parameter_sweep_int_pun_self)


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


##################
# Group size
##################

parameter_10 = SimulationParameters(action0=0.1f0,
                                       norm0=2.0f0,
                                       ext_pun0=0.1f0,
                                       int_pun_ext0=0.0f0,
                                       int_pun_self0=0.0f0,
                                       population_size=50,
                                       group_size=10,
                                       relatedness=0.5);
population_10 = population_construction(parameter_10);
@time simulation_10 = simulation(population_10);
@profview @time simulation_10 = simulation(population_10);


parameter_20 = SimulationParameters(action0=0.1f0,
                                       norm0=2.0f0,
                                       ext_pun0=0.1f0,
                                       int_pun_ext0=0.0f0,
                                       int_pun_self0=0.0f0,
                                       population_size=50,
                                       group_size=20,
                                       relatedness=0.5);
population_20 = population_construction(parameter_20);
@time simulation_20 = simulation(population_20);
@profview @time simulation_20 = simulation(population_20);