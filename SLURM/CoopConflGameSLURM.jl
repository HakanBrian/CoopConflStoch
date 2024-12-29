using BenchmarkTools
@everywhere include("CoopConflGameSLURMHelper.jl")
addprocs(128);


###############################
# Run Simulation
###############################

run_sim_r("simulation_sweep_r_stats.csv")

run_sim_rep("simulation_sweep_rep_stats.csv")

run_sim_rgs("simulation_sweep_rgs_stats.csv")