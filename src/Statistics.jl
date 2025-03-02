module Statistics

export calculate_statistics,
    statistics_filtered,
    statistics_filtered_sweep,
    statistics_full

using ..MainSimulation.IOHandler
import ..MainSimulation.IOHandler: generate_filename_suffix

using DataFrames, StatsBase

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

function statistics_filtered(
    df::DataFrame,
    sweep_vars::Dict{Symbol,AbstractVector},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{<:Real}} = nothing,
)
    # Determine the number of parameters in df
    num_params = maximum(df.param_id)

    # Determine total number of generations
    total_generations = maximum(df.generation)

    # Handle empty save_generations by defaulting to the last generation
    save_generations = isnothing(save_generations) ? [total_generations] : save_generations

    # Extract absolute and percentage-based generations
    abs_generations = Int64.(filter(x -> x isa Int, save_generations))
    pct_generations = filter(x -> x isa Float64, save_generations)

    # Convert percentage-based values to absolute generations (ensuring they are Int64)
    percent_generations = Int64.([
        round(Int64, p * total_generations / output_save_tick) * output_save_tick for p in pct_generations
    ])

    # Combine absolute and percentage-based generations, ensuring uniqueness and Int64 type
    generations_to_select = unique(Int64.(vcat(abs_generations, percent_generations)))

    # Initialize a dictionary to store results
    filtered_data = Dict{String,DataFrame}()

    # Sort the keys alphabetically
    sorted_keys = sort(collect(keys(sweep_vars)))

    # Generate all parameter combinations in a vector of dictionaries
    param_combinations = vec([
        Dict(k => v for (k, v) in zip(sorted_keys, values))
        for values in Iterators.product((sweep_vars[k] for k in sorted_keys)...)
    ])

    # Ensure we have enough parameter sets
    if length(param_combinations) != num_params
        @warn "Mismatch: Generated $(length(param_combinations)) parameter sets but found $num_params unique params in df."
    end

    # Create a mapping from `param_id` to its parameter combination
    param_id_to_params = Dict(i => param_combinations[i] for i in 1:num_params)

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, df)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Ensure this param_id exists in our mapping
        if i ∉ keys(param_id_to_params)
            @warn "No parameter combination found for param_id $i, skipping."
            continue
        end
        param_dict = param_id_to_params[i]  # Correct mapping

        # Collect data for each specified generation
        for gen in generations_to_select
            # Filter data for this generation
            gen_data = filter(row -> row.generation == gen, statistics)

            if !isempty(gen_data)
                # Generate the filename suffix correctly
                key = generate_filename_suffix(param_dict, "Filtered", time_point=gen)

                # Ensure the key exists
                if !haskey(filtered_data, key)
                    filtered_data[key] = DataFrame()
                end

                # Append data
                append!(filtered_data[key], gen_data)
            else
                @warn "Generation $gen not found in data"
            end
        end
    end

    # Remove empty DataFrames
    filter!(kv -> !isempty(kv.second), filtered_data)

    return filtered_data
end

function statistics_filtered_sweep(
    df::DataFrame,
    sweep_vars::Dict{Symbol,AbstractVector},
    output_save_tick::Int,
    save_generations::Union{Nothing,Vector{<:Real}} = nothing,
)
    # Calculate statistics for each parameter combination
    statistics_data = statistics_filtered(df, sweep_vars, output_save_tick, save_generations)

    # Sort the keys alphabetically
    sorted_keys = sort(collect(keys(sweep_vars)))

    # Generate all parameter combinations in a vector of dictionaries
    parameters = vec([
        Dict(k => v for (k, v) in zip(sorted_keys, values))
        for values in Iterators.product((sweep_vars[k] for k in sorted_keys)...)
    ])

    # Convert `parameters` into a DataFrame
    param_df = DataFrame()
    for key in sorted_keys
        param_df[!, key] = getindex.(parameters, key)
    end

    # Process each DataFrame in the statistics dictionary
    for (key, df) in statistics_data
        # Remove `generation` column
        select!(df, Not(:generation))

        # Add parameter columns **before existing columns**
        df = hcat(param_df, df)  # Concatenates param_df with df
        statistics_data[key] = df  # Update the DataFrame in the dictionary
    end

    return statistics_data
end

function statistics_full(df::DataFrame, sweep_vars::Dict{Symbol,AbstractVector})
    # Determine the number of parameters
    num_params = maximum(df.param_id)

    # Initialize dictionary to store DataFrames for each parameter combination
    independent_data = Dict{String, DataFrame}()

    # Generate parameter combinations
    param_combinations = [
        Dict(k => v for (k, v) in zip(keys(sweep_vars), values)) 
        for values in Iterators.product(values(sweep_vars)...)
    ]

    # Ensure we have enough parameter sets
    if length(param_combinations) != num_params
        @warn "Mismatch: Generated $(length(param_combinations)) parameter sets but found $num_params unique params in df."
    end  

    # Create a mapping from `param_id` to its parameter combination
    param_id_to_params = Dict(i => param_combinations[i] for i in 1:num_params)

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, df)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Ensure this param_id exists in our mapping
        if i ∉ keys(param_id_to_params)
            @warn "No parameter combination found for param_id $i, skipping."
            continue
        end
        param_dict = param_id_to_params[i]  # Correct mapping

        # Generate the suffix in the same format as `generate_filename_suffix`
        key = generate_filename_suffix(param_dict, "Full")

        # Add data
        independent_data[key] = statistics
    end

    # Remove empty DataFrames
    filter!(kv -> !isempty(kv.second), independent_data)

    return independent_data
end

end # module Statistics
