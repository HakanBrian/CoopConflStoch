using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


######################
# Behavior of BehavEq ###########################################################################################################
######################

base_params = MainSimulation.SimulationParameter(
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
parameter_sweep_action = [
    MainSimulation.update_params(base_params, action0 = action_value) for
    action_value in action_values
]
parameter_sweep_norm = [
    MainSimulation.update_params(base_params, norm0 = norm_value) for
    norm_value in norm_values
]
parameter_sweep_ext_pun = [
    MainSimulation.update_params(base_params, ext_pun0 = ext_pun_value) for
    ext_pun_value in ext_pun_values
]
parameter_sweep_int_pun_ext = [
    MainSimulation.update_params(base_params, int_pun_ext0 = int_pun_ext_value) for
    int_pun_ext_value in int_pun_ext_values
]
parameter_sweep_int_pun_self = [
    MainSimulation.update_params(base_params, int_pun_self0 = int_pun_self_value) for
    int_pun_self_value in int_pun_self_values
]

# Test the behavior of the behavioral equilibrium
function test_behav_eq(param_sweep::Vector{MainSimulation.SimulationParameter})
    actions = Vector{Float32}(undef, length(param_sweep))

    for (i, param) in enumerate(param_sweep)
        population = MainSimulation.population_construction(param)
        groups = MainSimulation.shuffle_and_group(
            population.groups,
            param.population_size,
            param.group_size,
            param.relatedness,
        )
        norm_pool, pun_pool = MainSimulation.collect_group(groups[1, :], population)
        action_sqrt = sqrt_llvm.(population.action)
        buffer = Vector{Float32}(undef, param.group_size - 1)
        MainSimulation.behavioral_equilibrium!(
            groups[1, :],
            buffer,
            action_sqrt,
            norm_pool,
            pun_pool,
            population,
        )
        actions[i] = mean(population.action)
    end

    p = MainSimulation.Plots.plot(
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