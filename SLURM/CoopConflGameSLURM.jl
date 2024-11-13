using Distributed
addprocs(4);

using BenchmarkTools
@everywhere include("../CoopConflGamePlots.jl")


r1_values = collect(range(0, 1.0, step=0.01));

parameter_sweep_r = [
    SimulationParameters(action0=0.1f0,
                         norm0=2.0f0,
                         ext_pun0=0.1f0,
                         int_pun_ext0=0.0f0,
                         int_pun_self0=0.0f0,
                         population_size=50,
                         group_size=10,
                         relatedness=r_values)
    for r_values in r1_values
]

simulation_sweep_r = simulation_replicate(parameter_sweep_r, 40);

save_simulation(simulation_sweep_r, joinpath(@__DIR__, "simulation_sweep_r1.csv"))


r05_values = collect(range(0, 0.5, step=0.01));
ep05_values = collect(range(0, 0.5, step=0.01));

parameter_sweep_rep = [
    SimulationParameters(action0=0.1f0,
                         norm0=2.0f0,
                         ext_pun0=ep_values,
                         int_pun_ext0=0.0f0,
                         int_pun_self0=0.0f0,
                         population_size=50,
                         group_size=10,
                         relatedness=r_values,
                         ext_pun_mutation_enabled=false)
    for r_values in r05_values
    for ep_values in ep05_values
]

simulation_sweep_rep = simulation_replicate(parameter_sweep_rep, 40);

save_simulation(simulation_sweep_rep, joinpath(@__DIR__, "simulation_sweep_rep1.csv"))