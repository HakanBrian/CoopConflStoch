using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, run_sim_r


###############################
# Run Simulation
###############################

base_params_gs_5 = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 5,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)
run_sim_r(
    base_params_gs_5,
    "data/rDiffGS/r2_gs_5",
    save_generations = [0.25, 0.5, 0.75, 1.0],
)


base_params_gs_50 = update_params(base_params_gs_5, group_size = 50);
run_sim_r(
    base_params_gs_50,
    "data/rDiffGS/r2_gs_50",
    save_generations = [0.25, 0.5, 0.75, 1.0],
)


base_params_gs_500 = update_params(base_params_gs_5, group_size = 500);
run_sim_r(
    base_params_gs_500,
    "data/rDiffGS/r1_gs_500",
    save_generations = [0.25, 0.5, 0.75, 1.0],
)
