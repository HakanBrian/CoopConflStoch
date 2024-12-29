using Distributed


####################################
# Helper Functions
####################################

@everywhere include("../CoopConflGamePlots.jl")


####################################
# SLURM Functions
####################################

function update_params(base_params::SimulationParameters; kwargs...)
    return SimulationParameters(; merge(Dict(fieldname => getfield(base_params, fieldname) for fieldname in fieldnames(SimulationParameters)), kwargs)...)
end

function run_sim_r(base_params::SimulationParameters, filename::String)
    r_values = collect(range(0, 1.0, step=0.01));

    parameter_sweep = [
        update_params(base_params, relatedness=r_value)
        for r_value in r_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end

function run_sim_rep(base_params::SimulationParameters, filename::String)
    r_values = collect(range(0, 0.5, step=0.05));
    ep_values = collect(range(0.0f0, 0.5f0, step=0.05f0));

    parameter_sweep = [
        update_params(base_params, relatedness=r_value, ext_pun0=ep_value)
        for r_value in r_values
        for ep_value in ep_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values, ep_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end

function run_sim_rgs(base_params::SimulationParameters, filename::String)
    r_values = collect(range(0, 0.5, step=0.05));
    gs_values = collect(range(50, 500, step=50));

    parameter_sweep = [
        update_params(base_params, relatedness=r_value, group_size=gs_value)
        for r_value in r_values
        for gs_value in gs_values
    ]

    simulation_sweep = simulation_replicate(parameter_sweep, 40);
    simulation_sweep_stats = sweep_statistics(simulation_sweep, r_values, gs_values)

    save_simulation(simulation_sweep_stats, joinpath(@__DIR__, filename))

    # Clear memory
    GC.gc()
end