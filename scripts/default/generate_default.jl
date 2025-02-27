using Distributed


###############################
# Game
###############################

cd(dirname(@__FILE__))  # change to the directory of this file
@everywhere include("../../src/processing.jl");


###############################
# Run Simulation
###############################

base_params = SimulationParameters(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 10,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)

run_sim_r(base_params, "r2.csv")

run_sim_rep(base_params, "rep2.csv")

run_sim_rip(base_params, "rep2.csv")

run_sim_rgs(base_params, "rgs2.csv")
