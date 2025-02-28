module RunSimulations

export run_sim_all, run_sim_r, run_sim_rep, run_sim_rip, run_sim_rgs

using ..MainSimulation.SimulationParameters
import ..MainSimulation.SimulationParameters: SimulationParameter, update_params

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population, population_construction

using ..MainSimulation.IOHandler
import ..MainSimulation.IOHandler: save_simulation, add_suffix_to_filepath

using ..MainSimulation.Simulations
import ..MainSimulation.Simulations: simulation

using ..MainSimulation.Statistics
import ..MainSimulation.Statistics:
    calculate_statistics,
    sweep_statistics_r,
    sweep_statistics_rep,
    sweep_statistics_rip,
    sweep_statistics_rgs,
    statistics_all

using Distributed, DataFrames, StatsBase


########################
# Simulation Replication ########################################################################################################
########################

function run_simulation(
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
        :a => mean,
        :p => mean,
        :T_ext => mean,
        :T_self => mean,
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
        run_simulation(parameters, 1, i)
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
        run_simulation(parameters, param_idx, replicate)
    end

    # Concatenate all results into a single DataFrame
    df = vcat(results...)

    return df
end


########
# SLURM #########################################################################################################################
########

function run_sim_all(
    base_params::SimulationParameter;
    num_replicates::Int = 40,
    save_file::Bool = true,
    filename::Union{String,Nothing} = nothing,
    sweep_full::Bool = false,
    sweep_vars::Union{Dict{Symbol,AbstractVector},Nothing} = nothing,
    statistics_function::Union{Function,Nothing} = nothing,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    # Ensure filename is provided if saving is enabled
    if save_file && isnothing(filename)
        error("A filename must be provided if save_file is true.")
    end

    # Determine if a sweep is being performed
    is_sweep = !isnothing(sweep_vars) && !isempty(sweep_vars)

    # Ensure sweep_vars and statistics_function are valid when sweeping
    if is_sweep
        if !sweep_full && isnothing(statistics_function)
            error("A statistics function must be provided when performing a sweep.")
        end

        # Generate parameter sweep
        parameters = vec([
            update_params(base_params; NamedTuple{Tuple(keys(sweep_vars))}(values)...)
            for values in Iterators.product(values(sweep_vars)...)
        ])
    else
        # Wrap base_params in an array to maintain consistency
        parameters = [base_params]
    end

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameters, num_replicates)

    # Compute statistics dynamically
    simulation_sweep_stats = if is_sweep && !sweep_full
        statistics_function(
            simulation_sweep,
            values(sweep_vars)...,
            base_params.output_save_tick,
            save_generations,
        )
    elseif is_sweep && sweep_full
        statistics_all(simulation_sweep, sweep_vars)
    else
        calculate_statistics(simulation_sweep)
    end

    # Save results if needed
    if save_file
        for (key, df) in simulation_sweep_stats
            filepath = add_suffix_to_filepath(filename, key)
            save_simulation(df, filepath)
        end
    else
        return simulation_sweep_stats
    end

    # Clear memory
    GC.gc()
end

run_sim_r(
    base_params::SimulationParameter,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    filename = filename,
    save_file = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
    ),
    statistics_function = sweep_statistics_r,
    save_generations = save_generations,
)

run_sim_rep(
    base_params::SimulationParameter,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    filename = filename,
    save_file = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
    statistics_function = sweep_statistics_rep,
    save_generations = save_generations,
)

run_sim_rip(
    base_params::SimulationParameter,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    filename = filename,
    save_file = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :int_pun_ext0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
        :int_pun_self0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
    statistics_function = sweep_statistics_rip,
    save_generations = save_generations,
)

run_sim_rgs(
    base_params::SimulationParameter,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    num_replicates = 20,
    save_file = true,
    filename = filename,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.1)),
        :group_size =>
            [collect(range(5, 50, step = 5))..., collect(range(50, 500, step = 50))...],
    ),
    statistics_function = sweep_statistics_rgs,
    save_generations = save_generations,
)

end # module RunSimulations
