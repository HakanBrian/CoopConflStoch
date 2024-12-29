using BenchmarkTools
@everywhere include("CoopConflGameSLURMHelper.jl")
addprocs(128);


###############################
# Run Simulation
###############################

base_params = SimulationParameters(action0=0.1f0,
                                    norm0=2.0f0,
                                    ext_pun0=0.1f0,
                                    population_size=500)

run_sim_r(base_params, "simulation_sweep_r_stats.csv")

run_sim_rep(base_params, "simulation_sweep_rep_stats.csv")

run_sim_rgs(base_params, "simulation_sweep_rgs_stats.csv")