####################################
# SLURM Functions
####################################

function update_params(base_params::SimulationParameters; kwargs...)
    # Update parameters by merging base parameters with new parameters
    return SimulationParameters(;
        merge(
            Dict(
                fieldname => getfield(base_params, fieldname) for
                fieldname in fieldnames(SimulationParameters)
            ),
            kwargs,
        )...,
    )
end

function run_sim_r(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Vector{Int64} = Int[],
    percentages_to_save::Vector{Float64} = Float64[],
)
    # Sweep over relatedness values
    r_values = collect(range(0, 1.0, step = 0.01))

    # Generate parameter sweep
    parameter_sweep =
        [update_params(base_params, relatedness = r_value) for r_value in r_values]

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, 40)
    simulation_sweep_stats = sweep_statistics_r(
        simulation_sweep,
        r_values,
        base_params.output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Save simulation data
    for (key, df) in simulation_sweep_stats
        save_simulation(df, joinpath(@__DIR__, filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end

function run_sim_rep(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Vector{Int64} = Int[],
    percentages_to_save::Vector{Float64} = Float64[],
)
    # Sweep over relatedness and external punishment values
    r_values = collect(range(0, 0.5, step = 0.05))
    ep_values = collect(range(0.0f0, 0.5f0, step = 0.05f0))

    # Generate parameter sweep
    parameter_sweep = [
        update_params(
            base_params,
            relatedness = r_value,
            ext_pun0 = ep_value,
            ext_pun_mutation_enabled = false,
        ) for r_value in r_values for ep_value in ep_values
    ]

    # Run simulation ad calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, 40)
    simulation_sweep_stats = sweep_statistics_rep(
        simulation_sweep,
        r_values,
        ep_values,
        base_params.output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Save simulation data
    for (key, df) in simulation_sweep_stats
        save_simulation(df, joinpath(@__DIR__, filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end

function run_sim_rip(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Vector{Int64} = Int[],
    percentages_to_save::Vector{Float64} = Float64[],
)
    # Sweep over relatedness and internal punishment values
    r_values = collect(range(0, 0.5, step = 0.05))
    ip_values = collect(range(0.0f0, 0.5f0, step = 0.05f0))

    # Generate parameter sweep
    parameter_sweep = [
        update_params(
            base_params,
            relatedness = r_value,
            int_pun_ext0 = ip_value,
            int_pun_self0 = ip_value,
            int_pun_ext_mutation_enabled = false,
            int_pun_self_mutation_enabled = false,
        ) for r_value in r_values for ip_value in ip_values
    ]

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, 40)
    simulation_sweep_stats = sweep_statistics_rip(
        simulation_sweep,
        r_values,
        ip_values,
        base_params.output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Save simulation data
    for (key, df) in simulation_sweep_stats
        save_simulation(df, joinpath(@__DIR__, filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end

function run_sim_rgs(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Vector{Int64} = Int[],
    percentages_to_save::Vector{Float64} = Float64[],
)
    # Sweep over relatedness and group size values
    r_values = collect(range(0, 0.5, step = 0.05))
    gs_values = collect(range(50, 500, step = 50))

    # Generate parameter sweep
    parameter_sweep = [
        update_params(base_params, relatedness = r_value, group_size = gs_value) for
        r_value in r_values for gs_value in gs_values
    ]

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, 20)
    simulation_sweep_stats = sweep_statistics_rgs(
        simulation_sweep,
        r_values,
        gs_values,
        base_params.output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Save simulation data
    for (key, df) in simulation_sweep_stats
        save_simulation(df, joinpath(@__DIR__, filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end
