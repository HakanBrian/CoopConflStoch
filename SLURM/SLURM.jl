using Distributed
addprocs(128);


###############################
# Game Function
###############################

@everywhere include("../funcs.jl");
include("SLURM_helper.jl");


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

run_sim_r(base_params, "r1", Int[], Float64[])

run_sim_rep(base_params, "rep1", Int[], Float64[])

run_sim_rip(base_params, "rep1", Int[], Float64[])

run_sim_rgs(base_params, "rgs1", Int[], Float64[])
