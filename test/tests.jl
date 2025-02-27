using BenchmarkTools


#######
# Game ##########################################################################################################################
#######

include("../src/Simulations.jl")
using .Simulations


##########################
# Population Construction #######################################################################################################
##########################

params = Simulations.SimulationParameter()  # uses all default values
population = Simulations.population_construction(params);


###################################################
# BehavEq & Payoff & Fitness & Social Interactions ##############################################################################
###################################################

params = Simulations.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 10,
    group_size = 2,
    relatedness = 1.0,
)
population = Simulations.population_construction(params)

groups = Simulations.shuffle_and_group(
    population.groups,
    params.population_size,
    params.group_size,
    params.relatedness,
)
norm_pool = sum(@view population.norm[groups[1, :]]) / params.group_size
pun_pool = sum(@view population.ext_pun[groups[1, :]]) / params.group_size
action_sqrt = Simulations.sqrt_llvm.(population.action)
action_sqrt_view = view(action_sqrt, groups[1, :])
action_sqrt_sum = sum(@view action_sqrt[groups[1, :]])

# Calculate behav eq
@time Simulations.behavioral_equilibrium!(
    groups[1, :],
    action_sqrt,
    action_sqrt_sum,
    norm_pool,
    pun_pool,
    population,
)

# Calculate best response
@code_warntype Simulations.best_response(
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
Simulations.total_payoff!(groups[1, :], norm_pool, pun_pool, population)
println(population.payoff)

# Calculate fitness
Simulations.fitness(population, groups[1, 1])

# Run social interactions
@time Simulations.social_interactions!(population)
println(population)


######################
# Behavior of BehavEq ###########################################################################################################
######################

base_params = Simulations.SimulationParameter(
    action0 = 0.0f0,
    norm0 = 0.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 10,
    group_size = 10,
    relatedness = 0.0,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)

# Sweep over different parameter values
action_values = collect(range(0.0f0, 10.0f0, step = 0.01f0));
norm_values = collect(range(0.0f0, 10.0f0, step = 0.01f0));
ext_pun_values = collect(range(0.0f0, 10.0f0, step = 0.01f0));
int_pun_ext_values = collect(range(0.0f0, 10.0f0, step = 0.01f0));
int_pun_self_values = collect(range(0.0f0, 10.0f0, step = 0.01f0));

# Create parameter sweeps
parameter_sweep_action =
    [Simulations.update_params(base_params, action0 = action_value) for action_value in action_values]
parameter_sweep_norm =
    [Simulations.update_params(base_params, norm0 = norm_value) for norm_value in norm_values]
parameter_sweep_ext_pun = [
    Simulations.update_params(base_params, ext_pun0 = ext_pun_value) for ext_pun_value in ext_pun_values
]
parameter_sweep_int_pun_ext = [
    Simulations.update_params(base_params, int_pun_ext0 = int_pun_ext_value) for
    int_pun_ext_value in int_pun_ext_values
]
parameter_sweep_int_pun_self = [
    Simulations.update_params(base_params, int_pun_self0 = int_pun_self_value) for
    int_pun_self_value in int_pun_self_values
]

# Test the behavior of the behavioral equilibrium
function test_behav_eq(param_sweep::Vector{Simulations.SimulationParameters.SimulationParameter})
    actions = Vector{Float32}(undef, length(param_sweep))

    for (i, param) in enumerate(param_sweep)
        population = Simulations.population_construction(param)
        groups = Simulations.shuffle_and_group(
            population.groups,
            param.population_size,
            param.group_size,
            param.relatedness,
        )
        norm_pool, pun_pool = Simulations.collect_group(groups[1, :], population)
        action_sqrt = Simulations.sqrt_llvm.(population.action)
        buffer = Vector{Float32}(undef, param.group_size - 1)
        Simulations.behavioral_equilibrium!(
            groups[1, :],
            buffer,
            action_sqrt,
            norm_pool,
            pun_pool,
            population,
        )
        actions[i] = mean(population.action)
    end

    p = Simulations.Plots.plot(
        action_values,
        actions,
        legend = true,
        label = "Action",
        xlabel = "initial",
        ylabel = "Action",
        title = "Behavioral Equilibrium",
        size = (800, 600),
    )

    # Display the plot
    display("image/png", p)
end

# Run the tests
test_behav_eq(parameter_sweep_action)
test_behav_eq(parameter_sweep_norm)
test_behav_eq(parameter_sweep_ext_pun)
test_behav_eq(parameter_sweep_int_pun_ext)
test_behav_eq(parameter_sweep_int_pun_self)


############
# Reproduce #####################################################################################################################
############

# IMPORTANT: To use this test payoffs need to be copied into the next generation !!!

# Create sample population
param = Simulations.SimulationParameter(
    action0 = 0.5f0,
    norm0 = 0.5f0,
    ext_pun0 = 0.0f0,
    generations = 10,
    population_size = 1000,
    mutation_rate = 0.0,
)
population = Simulations.population_construction(param)
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
Simulations.reproduce!(population)

# Offspring should have parent 4 as their parent ~40% of the time (only if there is no scaling)
println(
    "New population with payoff 4: ",
    count(payoff -> payoff == 4.0f0, population.payoff),
)


#########
# Mutate ########################################################################################################################
#########

# Create test mutate function
Simulations.mutate!(population, Simulations.truncation_bounds(population.parameters.mutation_variance, 0.99))
Simulations.println(population)


############
# Profiling #####################################################################################################################
############

# compilation
@time Simulations.simulation(population);
# pure runtime
@profview @time Simulations.simulation(population);


#############
# Group size ####################################################################################################################
#############

# Test group size 10
parameter_10 = Simulations.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 50,
    group_size = 10,
    relatedness = 0.5,
);
population_10 = Simulations.population_construction(parameter_10);
@time simulation_10 = Simulations.simulation(population_10);
@profview @time simulation_10 = Simulations.simulation(population_10);

# Test group size 20
parameter_20 = Simulations.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 50,
    group_size = 20,
    relatedness = 0.5,
);
population_20 = Simulations.population_construction(parameter_20);
@time simulation_20 = Simulations.simulation(population_20);
@profview @time simulation_20 = Simulations.simulation(population_20);
