using Distributed


###############################
# Game
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, run_sim_all


###############################
# Run Simulation
###############################

sweep_r = Dict{Symbol,AbstractVector}(:relatedness => collect(range(0, 1.0, step = 0.25)));

base_param = SimulationParameters(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 200000,
    population_size = 500,
    group_size = 5,
    relatedness = 0.0,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 20,
)
run_sim_all(
    base_param,
    filename = "data/basin/basin_5",
    sweep_full = true,
    sweep_vars = sweep_r,
)


base_param_50 = update_params(base_param, group_size = 50)
run_sim_all(
    base_param_50,
    filename = "data/basin/basin_50",
    sweep_full = true,
    sweep_vars = sweep_r,
)


base_param_500 = update_params(base_param, group_size = 500)
run_sim_all(
    base_param_500,
    filename = "data/basin/basin_500",
    sweep_full = true,
    sweep_vars = sweep_r,
)
