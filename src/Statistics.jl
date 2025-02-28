module Statistics

export statistics_selection,
    sweep_statistics_r,
    sweep_statistics_rep,
    sweep_statistics_rip,
    sweep_statistics_rgs,
    statistics_all

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

end # module Statistics
