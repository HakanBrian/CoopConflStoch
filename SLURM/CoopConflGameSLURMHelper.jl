using Distributed


####################################
# Helper Functions
####################################

@everywhere include("../CoopConflGamePlots.jl")


####################################
# SLURM Functions
####################################

function run_sim_r(filename::String)
    r_values = collect(range(0, 1.0, step=0.01));

    parameter_sweep = [
        SimulationParameters(action0=0.1f0,
                            norm0=2.0f0,
                            ext_pun0=0.1f0,
                            int_pun_ext0=0.0f0,
                            int_pun_self0=0.0f0,
                            population_size=500,
                            group_size=10,
                            relatedness=r_value)
        for r_value in r_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end

function run_sim_rep(filename::String)
    r_values = collect(range(0, 0.5, step=0.05));
    ep_values = collect(range(0.0f0, 0.5f0, step=0.05f0));

    parameter_sweep = [
        SimulationParameters(action0=0.1f0,
                            norm0=2.0f0,
                            ext_pun0=ep_value,
                            int_pun_ext0=0.0f0,
                            int_pun_self0=0.0f0,
                            population_size=500,
                            group_size=10,
                            relatedness=r_value,
                            ext_pun_mutation_enabled=false)
        for r_value in r_values
        for ep_value in ep_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values, ep_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end

function run_sim_rgs(filename::String)
    r_values = collect(range(0, 0.5, step=0.05));
    gs_values = collect(range(50, 500, step=50));

    parameter_sweep = [
        SimulationParameters(action0=0.1f0,
                            norm0=2.0f0,
                            ext_pun0=0.1f0,
                            int_pun_ext0=0.0f0,
                            int_pun_self0=0.0f0,
                            population_size=500,
                            group_size=gs_value,
                            relatedness=r_value)
        for r_value in r_values
        for gs_value in gs_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values, gs_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end