##################
# SLURM Functions ###############################################################################################################
##################

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

function run_sim_sweep(
    base_params::SimulationParameters,
    filename::String,
    sweep_vars::Dict{Symbol, AbstractVector},
    statistics_function::Function,
    num_replicates::Int = 40,
    generations_to_save::Vector{Int64} = Int[],
    percentages_to_save::Vector{Float64} = Float64[]
)
    # Generate parameter sweep
    parameter_sweep = [
        update_params(base_params; NamedTuple{keys(sweep_vars)}(values)) for
        values in Iterators.product(values(sweep_vars)...)
    ]

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, num_replicates)
    simulation_sweep_stats = statistics_function(
        simulation_sweep,
        values(sweep_vars)...,
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

function run_sim_r(base_params::SimulationParameters, filename::String, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    run_sim_sweep(
        base_params,
        filename,
        Dict(:relatedness => collect(range(0, 1.0, step=0.01))),
        sweep_statistics_r,
        40,
        generations_to_save,
        percentages_to_save
    )
end

function run_sim_rep(base_params::SimulationParameters, filename::String, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    run_sim_sweep(
        base_params,
        filename,
        Dict(
            :relatedness => collect(range(0, 0.5, step=0.05)),
            :ext_pun0 => collect(range(0.0f0, 0.5f0, step=0.05f0))
        ),
        sweep_statistics_rep,
        40,
        generations_to_save,
        percentages_to_save
    )
end

function run_sim_rip(base_params::SimulationParameters, filename::String, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    run_sim_sweep(
        base_params,
        filename,
        Dict(
            :relatedness => collect(range(0, 0.5, step=0.05)),
            :int_pun_ext0 => collect(range(0.0f0, 0.5f0, step=0.05f0)),
            :int_pun_self0 => collect(range(0.0f0, 0.5f0, step=0.05f0))
        ),
        sweep_statistics_rip,
        40,
        generations_to_save,
        percentages_to_save
    )
end

function run_sim_rgs(base_params::SimulationParameters, filename::String, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    run_sim_sweep(
        base_params,
        filename,
        Dict(
            :relatedness => collect(range(0, 0.5, step=0.05)),
            :group_size => collect(range(50, 500, step=50))
        ),
        sweep_statistics_rgs,
        20,
        generations_to_save,
        percentages_to_save
    )
end
