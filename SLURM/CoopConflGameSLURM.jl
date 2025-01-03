using Distributed
addprocs(128);


###############################
# Game Function
###############################

@everywhere include("../CoopConflGameFuncs.jl");
include("CoopConflGameSLURMHelper.jl");


###############################
# Run Simulation
###############################

base_params = SimulationParameters(action0=0.1f0,
                                    norm0=2.0f0,
                                    ext_pun0=0.1f0,
                                    population_size=500)

run_sim_r(base_params, "r1.csv")

run_sim_rep(base_params, "rep1.csv")

run_sim_rgs(base_params, "rgs1.csv")