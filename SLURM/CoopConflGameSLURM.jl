using Distributed
addprocs(4);

using BenchmarkTools
@everywhere include(joinpath(@__DIR__, "../CoopConflGameFuncs.jl"))
#=
relatedness_values = collect(range(0, 1, step=0.01));

parameter_sweep = [
    SimulationParameters(action0=0.1f0,
                         norm0=2.0f0,
                         ext_pun0=0.1f0,
                         int_pun_ext0=0.0f0,
                         int_pun_self0=0.0f0,
                         population_size=50,
                         group_size=10,
                         relatedness=r_values)
    for r_values in relatedness_values
]

simulation_sweep = simulation_replicate(parameter_sweep, 40);
=#
param=SimulationParameters(action0=0.1f0,
norm0=2.0f0,
ext_pun0=0.1f0,
int_pun_ext0=0.0f0,
int_pun_self0=0.0f0,
population_size=50,
group_size=10,
relatedness=r_values)

simulation = simulation_replicate(param, 5);

save_simulation(simulation, joinpath(@__DIR__, "simulation.csv"))