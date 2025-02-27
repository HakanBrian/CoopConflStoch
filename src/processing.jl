using Distributed


#########
# Helper ########################################################################################################################
#########

include("funcs.jl")


########################
# Simulation Replication ########################################################################################################
########################

function run_simulation(
    parameters::SimulationParameters,
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

function simulation_replicate(parameters::SimulationParameters, num_replicates::Int64)
    # Use pmap to parallelize the simulation
    results = pmap(1:num_replicates) do i
        run_simulation(parameters, 1, i)
    end

    # Concatenate all the simulation means returned by each worker
    df = vcat(results...)

    return df
end

function simulation_replicate(
    parameter_sweep::Vector{SimulationParameters},
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


#############
# Statistics ####################################################################################################################
#############

function calculate_statistics(df::DataFrame)
    # Group by generation
    grouped = groupby(df, :generation)

    # Calculate mean and standard deviation for each trait across replicates
    stats = combine(
        grouped,
        :action_mean => mean => :action_mean_mean,
        :action_mean => std => :action_mean_std,
        :a_mean => mean => :a_mean_mean,
        :a_mean => std => :a_mean_std,
        :p_mean => mean => :p_mean_mean,
        :p_mean => std => :p_mean_std,
        :T_ext_mean => mean => :T_ext_mean_mean,
        :T_ext_mean => std => :T_ext_mean_std,
        :T_self_mean => mean => :T_self_mean_mean,
        :T_self_mean => std => :T_self_mean_std,
        :payoff_mean => mean => :payoff_mean_mean,
        :payoff_mean => std => :payoff_mean_std,
    )

    return stats
end

function statistics_selection(
    df::DataFrame,
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    # Determine the number of params
    num_params = maximum(df.param_id)

    # Determine total number of generations
    total_generations = maximum(df.generation)

    # Initialize a dictionary to store DataFrames for each specified generation or percentage
    selected_data = Dict{String,DataFrame}()

    # Handle empty save_generations by defaulting to the last generation
    save_generations = isnothing(save_generations) ? [total_generations] : save_generations

    # Extract absolute and percentage-based generations
    abs_generations = filter(x -> x isa Int, save_generations)
    pct_generations = filter(x -> x isa Float64, save_generations)

    # Convert percentage-based values to specific generations
    percent_generations = [
        (p, round(Int, p * total_generations / output_save_tick) * output_save_tick) for
        p in pct_generations
    ]

    # Combine absolute and percentage-based generations, ensuring uniqueness
    generations_to_select =
        unique(vcat(abs_generations, map(x -> x[2], percent_generations)))

    # Preallocating keys in dictionary
    for gen in generations_to_select
        selected_data[string("gen_", gen)] = DataFrame()
    end

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, df)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Collect rows corresponding to the specified generations
        for gen in generations_to_select
            # Retrieve the data associated with this generation
            gen_data = filter(row -> row.generation == gen, statistics)

            if !isempty(gen_data)
                # Determine key and append data
                key = string("gen_", gen)
                append!(selected_data[key], gen_data)
            else
                @warn "Generation $gen not found in data"
            end
        end
    end

    # Remove empty DataFrames from the dictionary efficiently
    filter!(kv -> !isempty(kv.second), selected_data)

    return selected_data
end

function sweep_statistics_r(
    df::DataFrame,
    r_values::Vector{Float64},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    statistics_r = statistics_selection(df, output_save_tick, save_generations)

    # Add relatedness columns to each DataFrame
    for (key, df) in statistics_r
        rename!(df, :generation => :relatedness)
        df.relatedness = r_values
    end

    return statistics_r
end

function sweep_statistics_rep(
    df::DataFrame,
    r_values::Vector{Float64},
    ep_values::Vector{Float32},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    statistics_rep = statistics_selection(df, output_save_tick, save_generations)

    # Add relatedness and ext_pun columns to each DataFrame
    for (key, df) in statistics_rep
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(ep_values))

        insertcols!(df, 2, :ext_pun => repeat(ep_values, length(r_values)))
        select!(df, Not([:p_mean_mean, :p_mean_std]))
    end

    return statistics_rep
end

function sweep_statistics_rip(
    df::DataFrame,
    r_values::Vector{Float64},
    ip_values::Vector{Float32},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    statistics_rip = statistics_selection(df, output_save_tick, save_generations)

    # Add relatedness and int_pun columns to each DataFrame
    for (key, df) in statistics_rip
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(ip_values))

        insertcols!(df, 2, :int_pun => repeat(ip_values, length(r_values)))
        select!(
            df,
            Not([:T_ext_mean_mean, :T_ext_mean_std, :T_self_mean_mean, :T_self_mean_std]),
        )
    end

    return statistics_rip
end

function sweep_statistics_rgs(
    df::DataFrame,
    r_values::Vector{Float64},
    gs_values::Vector{Int64},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{Real}} = nothing,
)
    statistics_rgs = statistics_selection(df, output_save_tick, save_generations)

    # Add relatedness and group_size columns to each DataFrame
    for (key, df) in statistics_rgs
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(gs_values))

        insertcols!(df, 2, :group_size => repeat(gs_values, length(r_values)))
    end

    return statistics_rgs
end

function statistics_all(df::DataFrame, sweep_var::Dict{Symbol,AbstractVector})
    # Determine the number of params
    num_params = maximum(df.param_id)

    # Extract the independent variable
    independent_var = only(keys(sweep_var))

    # Initialize a dictionary to store DataFrames for each independent value
    independent_data = Dict{String,DataFrame}()

    # Iterate over the values of the independent variable
    for value in sweep_var[independent_var]
        independent_data[string(independent_var, "_", value)] = DataFrame()
    end

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, df)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Collect data corresponding to the specified independent variable values
        key = string(independent_var, "_", sweep_var[independent_var][i])
        append!(independent_data[key], statistics)
    end

    # Remove empty DataFrames from the dictionary efficiently
    filter!(kv -> !isempty(kv.second), independent_data)

    return independent_data
end


########
# SLURM #########################################################################################################################
########

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

function run_sim_all(
    base_params::SimulationParameters;
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
        parameters = [
            update_params(base_params; NamedTuple{Tuple(keys(sweep_vars))}(values)...)
            for values in Iterators.product(values(sweep_vars)...)
        ]
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
            filepath = joinpath(pwd(), filename * "_" * key * ".csv")
            save_simulation(df, filepath)
        end
    else
        return simulation_sweep_stats
    end

    # Clear memory
    GC.gc()
end

run_sim_r(
    base_params::SimulationParameters,
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
    base_params::SimulationParameters,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    filename = filename,
    save_file = true,
    sweep_vars = Dict(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
    statistics_function = statistics_selection,
    save_generations = save_generations,
)

run_sim_rip(
    base_params::SimulationParameters,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    filename = filename,
    save_file = true,
    sweep_vars = Dict(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :int_pun_ext0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
        :int_pun_self0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
    statistics_function = sweep_statistics_rip,
    save_generations = save_generations,
)

run_sim_rgs(
    base_params::SimulationParameters,
    filename::String;
    save_generations::Union{Nothing,Vector{Real}} = nothing,
) = run_sim_all(
    base_params,
    num_replicates = 20,
    save_file = true,
    filename = filename,
    sweep_vars = Dict(
        :relatedness => collect(range(0, 1.0, step = 0.1)),
        :group_size => [collect(range(5, 50, step = 5))..., collect(range(50, 500, step = 50))...],
    ),
    statistics_function = sweep_statistics_rgs,
    save_generations = save_generations,
)
