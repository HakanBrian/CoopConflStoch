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
    all_simulation_means = vcat(results...)

    return all_simulation_means
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
    all_simulation_means = vcat(results...)

    return all_simulation_means
end


#############
# Statistics ####################################################################################################################
#############

function calculate_statistics(all_simulation_means::DataFrame)
    # Group by generation
    grouped = groupby(all_simulation_means, :generation)

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
    all_simulation_means::DataFrame,
    output_save_tick::Int,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

    # Determine total number of generations
    total_generations = maximum(all_simulation_means.generation)

    # Initialize a dictionary to store DataFrames for each specified generation or percentage
    selected_data = Dict{String,DataFrame}()

    # Initialize empty arrays if no specific generations or percentages are given
    generations_to_save = isnothing(generations_to_save) ? Int64[] : generations_to_save
    percentages_to_save = isnothing(percentages_to_save) ? Float64[] : percentages_to_save

    # Calculate the specific generations based on percentages
    percent_generations = [
        (p, round(Int, p * total_generations / output_save_tick) * output_save_tick) for
        p in percentages_to_save
    ]

    # Combine specific generations and percentage-based generations
    generations_to_select =
        unique(vcat(generations_to_save, map(x -> x[2], percent_generations)))

    # If no specific generations or percentages are given, save the last row
    if isempty(generations_to_select)
        generations_to_select = [total_generations]  # Default to the last generation
        generations_to_save = [total_generations]
    end

    # Preallocating keys in dictionary
    for gen in generations_to_select
        selected_data[string("gen_", gen)] = DataFrame()
    end

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, all_simulation_means)

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
            elseif isempty(gen_data)
                @warn "Generation $gen not found in data"
            end
        end
    end

    # Remove empty DataFrames from the dictionary efficiently
    filter!(kv -> !isempty(kv.second), selected_data)

    return selected_data
end

function sweep_statistics_r(
    all_simulation_means::DataFrame,
    r_values::Vector{Float64},
    output_save_tick::Int,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    statistics_r = statistics_selection(
        all_simulation_means,
        output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Add relatedness columns to each DataFrame
    for (key, df) in statistics_r
        rename!(df, :generation => :relatedness)
        df.relatedness = r_values
    end

    return statistics_r
end

function sweep_statistics_rep(
    all_simulation_means::DataFrame,
    r_values::Vector{Float64},
    ep_values::Vector{Float32},
    output_save_tick::Int,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    statistics_rep = statistics_selection(
        all_simulation_means,
        output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

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
    all_simulation_means::DataFrame,
    r_values::Vector{Float64},
    ip_values::Vector{Float32},
    output_save_tick::Int,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    statistics_rip = statistics_selection(
        all_simulation_means,
        output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

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
    all_simulation_means::DataFrame,
    r_values::Vector{Float64},
    gs_values::Vector{Int64},
    output_save_tick::Int,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    statistics_rgs = statistics_selection(
        all_simulation_means,
        output_save_tick,
        generations_to_save,
        percentages_to_save,
    )

    # Add relatedness and group_size columns to each DataFrame
    for (key, df) in statistics_rgs
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(gs_values))

        insertcols!(df, 2, :group_size => repeat(gs_values, length(r_values)))
    end

    return statistics_rgs
end

function statistics_all(
    all_simulation_means::DataFrame,
    sweep_var::Dict{Symbol,AbstractVector},
)
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

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
        param_data = filter(row -> row.param_id == i, all_simulation_means)

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

function run_sim_sweep(
    base_params::SimulationParameters,
    filename::String,
    sweep_vars::Dict{Symbol,AbstractVector},
    statistics_function::Function,
    num_replicates::Int = 40,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    # Generate parameter sweep
    parameter_sweep = [
        update_params(base_params; NamedTuple{Tuple(keys(sweep_vars))}(values)...) for
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
        save_simulation(df, joinpath(pwd(), filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end

function run_sim_r(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    sweep_vars =
        Dict{Symbol,AbstractVector}(:relatedness => collect(range(0, 1.0, step = 0.01)))

    run_sim_sweep(
        base_params,
        filename,
        sweep_vars,
        sweep_statistics_r,
        40,
        generations_to_save,
        percentages_to_save,
    )
end

function run_sim_rep(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    sweep_vars = Dict(
        :relatedness => collect(range(0, 0.5, step = 0.05)),
        :ext_pun0 => collect(range(0.0f0, 0.5f0, step = 0.05f0)),
    )

    run_sim_sweep(
        base_params,
        filename,
        sweep_vars,
        sweep_statistics_rep,
        40,
        generations_to_save,
        percentages_to_save,
    )
end

function run_sim_rip(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    sweep_vars = Dict(
        :relatedness => collect(range(0, 0.5, step = 0.05)),
        :int_pun_ext0 => collect(range(0.0f0, 0.5f0, step = 0.05f0)),
        :int_pun_self0 => collect(range(0.0f0, 0.5f0, step = 0.05f0)),
    )

    run_sim_sweep(
        base_params,
        filename,
        sweep_vars,
        sweep_statistics_rip,
        40,
        generations_to_save,
        percentages_to_save,
    )
end

function run_sim_rgs(
    base_params::SimulationParameters,
    filename::String,
    generations_to_save::Union{Nothing,Vector{Int64}} = nothing,
    percentages_to_save::Union{Nothing,Vector{Float64}} = nothing,
)
    sweep_vars = Dict(
        :relatedness => collect(range(0, 0.5, step = 0.05)),
        :group_size => collect(range(50, 500, step = 50)),
    )

    run_sim_sweep(
        base_params,
        filename,
        sweep_vars,
        sweep_statistics_rgs,
        20,
        generations_to_save,
        percentages_to_save,
    )
end

function run_sim_all(
    base_params::SimulationParameters,
    filename::String,
    sweep_var::Dict{Symbol,AbstractVector},
    num_replicates::Int = 40,
)
    # Generate parameter sweep
    parameter_sweep = [
        update_params(base_params; NamedTuple{Tuple(keys(sweep_var))}(values)...) for
        values in Iterators.product(values(sweep_var)...)
    ]

    # Run simulation and calculate statistics
    simulation_sweep = simulation_replicate(parameter_sweep, num_replicates)
    simulation_sweep_stats = statistics_all(simulation_sweep, sweep_var)

    # Save simulation data
    for (key, df) in simulation_sweep_stats
        save_simulation(df, joinpath(pwd(), filename * "_" * key * ".csv"))
    end

    # Clear memory
    GC.gc()
end
