module RunSimulations

export run_simulation

using ..MainSimulation.SimulationParameters
import ..MainSimulation.SimulationParameters: SimulationParameter, generate_params

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population, population_construction

using ..MainSimulation.IOHandler
import ..MainSimulation.IOHandler: save_simulation, generate_filename_suffix, modify_filename

using ..MainSimulation.Simulations
import ..MainSimulation.Simulations: simulation

using ..MainSimulation.Statistics
import ..MainSimulation.Statistics:
    statistics_processed, statistics_filtered_processed, statistics_full

using Distributed, DataFrames, StatsBase


########################
# Simulation Replication ########################################################################################################
########################

function run_replicate(
    parameters::SimulationParameter,
    param_id::Int64,
    replicate_id::Int64,
)
    println("Running simulation replicate $replicate_id for param_id $param_id")

    # Run the simulation
    population = population_construction(parameters)
    simulation_replicate = simulation(population)

    # Group by generation and compute mean for each generation
    simulation_gdf = groupby(simulation_replicate, :generation)
    simulation_mean = combine(
        simulation_gdf,
        :action => mean,
        :norm => mean,
        :ext_pun => mean,
        :int_pun_ext => mean,
        :int_pun_self => mean,
        :payoff => mean,
    )

    # Add columns for replicate and param_id
    rows_to_insert = nrow(simulation_mean)
    insertcols!(simulation_mean, 1, :param_id => fill(param_id, rows_to_insert))
    insertcols!(simulation_mean, 2, :replicate => fill(replicate_id, rows_to_insert))

    return simulation_mean
end

function simulation_replicate(parameters::SimulationParameter, num_replicates::Int64)
    # Use pmap to parallelize the simulation
    results = pmap(1:num_replicates) do i
        run_replicate(parameters, 1, i)
    end

    # Concatenate all the simulation means returned by each worker
    df = vcat(results...)

    return df
end

function simulation_replicate(
    parameter_sweep::Vector{SimulationParameter},
    num_replicates::Int64,
)
    # Create a list of tasks (parameter set index, parameter set, replicate) to distribute
    tasks = [
        (idx, parameters, replicate) for (idx, parameters) in enumerate(parameter_sweep) for
        replicate in 1:num_replicates
    ]

    # Use pmap to distribute the tasks across the workers
    results = pmap(tasks) do task
        param_idx, parameters, replicate = task
        # Run simulation and store the result with the parameter set index
        run_replicate(parameters, param_idx, replicate)
    end

    # Concatenate all results into a single DataFrame
    df = vcat(results...)

    return df
end


########
# SLURM #########################################################################################################################
########

function run_simulation(
    base_params::SimulationParameter;
    num_replicates::Int = 40,
    save_file::Bool = true,
    filepath::Union{String,Nothing} = nothing,
    sweep_full::Bool = false,
    sweep_vars::Union{Dict{Symbol,Vector{<:Real}},Nothing} = nothing,
    linked_params::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
    save_generations::Union{Nothing,Vector{<:Real}} = nothing,
)
    # Ensure filepath is provided if saving is enabled
    if save_file && isnothing(filepath)
        error("A filepath must be provided if save_file is true.")
    end

    # Determine if a sweep is being performed
    is_sweep = !isnothing(sweep_vars) && !isempty(sweep_vars)

    # Ensure sweep_vars is valid when sweeping
    if is_sweep
        # Generate parameter sweep
        parameters = generate_params(base_params, sweep_vars, linked_params)
    else
        # Wrap base_params to maintain consistency
        parameters = base_params
    end

    # Run simulation and calculate statistics
    simulation_data = simulation_replicate(parameters, num_replicates)

    # Compute statistics dynamically
    simulation_sweep_stats = if is_sweep && !sweep_full
        statistics_filtered_processed(
            simulation_data,
            sweep_vars,
            base_params.output_save_tick,
            save_generations,
        )
    elseif is_sweep && sweep_full
        statistics_full(simulation_data, sweep_vars)
    else
        statistics_processed(simulation_data, parameters)
    end

    # Save results if needed
    if save_file
        if base_params.use_bipenal
            filename_pun = modify_filename(filepath, "bipenal")
        else
            filename_pun = modify_filename(filepath, "unipenal")
        end

        for (key, df) in simulation_sweep_stats
            filename_full = modify_filename(filename_pun, key)
            save_simulation(df, filename_full)
        end
    else
        return simulation_sweep_stats
    end

    # Clear memory
    GC.gc()
end

end # module RunSimulations
