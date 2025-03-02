using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation:
    SimulationParameter, run_sim_r, run_sim_rep, run_sim_rip, run_sim_rgs


###############################
# Run Simulation
###############################

base_params = SimulationParameter(
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

run_sim_r(base_params, "data/default/r2.csv")

run_sim_rep(base_params, "data/default/rep2.csv")

run_sim_rip(base_params, "data/default/rip2.csv")

run_sim_rgs(base_params, "data/default/rgs2.csv")
