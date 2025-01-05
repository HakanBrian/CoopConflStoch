using CSV, FilePathsBase


####################################
# Helper Functions
####################################

include("CoopConflGameStructs.jl")


###############################
# Population Simulation
###############################

function offspring!(pop::Population, offspring_index::Int64, parent_index::Int64)
    # Copy traits from parent to offspring
    pop.action[offspring_index] = pop.action[parent_index]
    pop.norm[offspring_index] = pop.norm[parent_index]
    pop.ext_pun[offspring_index] = pop.ext_pun[parent_index]
    pop.int_pun_ext[offspring_index] = pop.int_pun_ext[parent_index]
    pop.int_pun_self[offspring_index] = pop.int_pun_self[parent_index]

    # Set initial values for offspring
    pop.payoff[offspring_index] = 0.0f0
    pop.interactions[offspring_index] = 0
end

function truncation_bounds(variance::Float64, retain_proportion::Float64)
    # Calculate tail probability alpha
    alpha = 1 - retain_proportion

    # Calculate z-score corresponding to alpha/2
    z_alpha_over_2 = quantile(Normal(), 1 - alpha/2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * √variance
    upper_bound = z_alpha_over_2 * √variance

    return SA[lower_bound, upper_bound]
end


##################
# Behavioral Equilibrium
##################

function filter_out_val!(arr::AbstractVector{T}, exclude_val::T, buffer::Vector{T}) where T
    count = 1
    for i in eachindex(arr)
        if arr[i] != exclude_val  # Exclude based on the value
            buffer[count] = arr[i]
            count += 1
        end
    end
    return @inbounds view(buffer, 1:count-1)  # Return a view of the filtered buffer
end

function filter_out_idx!(arr::AbstractVector{T}, exclude_idx::Int, buffer::Vector{T}) where T
    count = 1
    for i in eachindex(arr)
        if i != exclude_idx  # Exclude based on the index
            buffer[count] = arr[i]
            count += 1
        end
    end
    return @inbounds view(buffer, 1:count-1)  # Return a view of the filtered buffer
end


##################
# Social Interaction
##################

function probabilistic_round(x::Float64)::Int64
    lower = floor(Int64, x)
    upper = ceil(Int64, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

function in_place_sample!(data::AbstractVector{T}, k::Int) where T
    n = length(data)

    for i in 1:k
        j = rand(i:n)  # Random index between i and n (inclusive)
        data[i], data[j] = data[j], data[i]  # Swap elements
    end
    return @inbounds view(data, 1:k)  # Return a view of the first k elements
end

function collect_group(group::AbstractVector{Int64}, pop::Population)
    norm_pool = sum(@view pop.norm[group]) / pop.parameters.group_size
    pun_pool = sum(@view pop.ext_pun[group]) / pop.parameters.group_size

    return norm_pool, pun_pool
end


##################
# Reproduction
##################

function normalize_exponentials(values::Vector{Exponential})
    max_base = maximum(v.base for v in values)
    shifted = [v.base - max_base for v in values]  # Shift to avoid large exponents
    probs = exp.(shifted)
    return probs ./ sum(probs)  # Normalize to probabilities
end


##################
# Statistics Function
##################

function calculate_statistics(all_simulation_means::DataFrame)
    # Group by generation
    grouped = groupby(all_simulation_means, :generation)

    # Calculate mean and standard deviation for each trait across replicates
    stats = combine(grouped,
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
                    :payoff_mean => std => :payoff_mean_std)

    return stats
end

function statistics_selection(all_simulation_means::DataFrame, output_save_tick::Int, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

    # Determine total number of generations
    total_generations = maximum(all_simulation_means.generation)

    # Initialize a dictionary to store DataFrames for each specified generation or percentage
    selected_data = Dict{String, DataFrame}()

    # Calculate the specific generations based on percentages
    percent_generations = [
        (p, round(Int, p * total_generations / output_save_tick) * output_save_tick)
        for p in percentages_to_save
    ]

    # Combine specific generations and percentage-based generations
    generations_to_select = unique(vcat(
        generations_to_save,
        map(x -> x[2], percent_generations)
    ))

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

    # Remove empty DataFrames from the dictionary
    for key in keys(selected_data)
        if isempty(selected_data[key])
            @warn "Key $key has an empty DataFrame and will be removed"
            delete!(selected_data, key)
        end
    end

    return selected_data
end

function sweep_statistics_r(all_simulation_means::DataFrame, r_values::Vector{Float64}, output_save_tick::Int, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    statistics_r = statistics_selection(all_simulation_means, output_save_tick, generations_to_save, percentages_to_save)

    # Add relatedness columns to each DataFrame
    for (key, df) in statistics_r
        rename!(df, :generation => :relatedness)
        df.relatedness = r_values
    end

    return statistics_r
end

function sweep_statistics_rep(all_simulation_means::DataFrame, r_values::Vector{Float64}, ep_values::Vector{Float32}, output_save_tick::Int, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    statistics_rep = statistics_selection(all_simulation_means, output_save_tick, generations_to_save, percentages_to_save)

    # Add relatedness and ext_pun columns to each DataFrame
    for (key, df) in statistics_rep
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(ep_values))

        insertcols!(df, 2, :ext_pun => repeat(ep_values, length(r_values)))
        select!(df, Not([:p_mean_mean, :p_mean_std]))
    end

    return statistics_rep
end

function sweep_statistics_rip(all_simulation_means::DataFrame, r_values::Vector{Float64}, ip_values::Vector{Float32}, output_save_tick::Int, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    statistics_rip = statistics_selection(all_simulation_means, output_save_tick, generations_to_save, percentages_to_save)

    for (key, df) in statistics_rip
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(ip_values))

        insertcols!(df, 2, :int_pun => repeat(ip_values, length(r_values)))
        select!(df, Not([:T_ext_mean_mean, :T_ext_mean_std, :T_self_mean_mean, :T_self_mean_std]))
    end

    return statistics_rip
end

function sweep_statistics_rgs(all_simulation_means::DataFrame, r_values::Vector{Float64}, gs_values::Vector{Int64}, output_save_tick::Int, generations_to_save::Vector{Int64} = Int[], percentages_to_save::Vector{Float64} = Float64[])
    statistics_rgs = statistics_selection(all_simulation_means, output_save_tick, generations_to_save, percentages_to_save)

    for (key, df) in statistics_rgs
        rename!(df, :generation => :relatedness)
        df.relatedness = repeat(r_values, inner = length(gs_values))

        insertcols!(df, 2, :group_size => repeat(gs_values, length(r_values)))
    end

    return statistics_rgs
end


##################
# I/O Function
##################

function save_simulation(simulation::DataFrame, filepath::String)
    # Ensure the filepath has the .csv extension
    if !endswith(filepath, ".csv")
        filepath *= ".csv"
    end

    # Convert to an absolute path (in case it's not already)
    filepath = abspath(filepath)

    # Check if the file already exists, and print a warning if it does
    if isfile(filepath)
        println("Warning: File '$filepath' already exists and will be overwritten.")
    end

    # Save the dataframe, overwriting the file if it exists
    CSV.write(filepath, simulation)
    println("File saved as: $filepath")
end

function read_simulation(filepath::String)
    # Ensure the filepath has the .csv extension
    if !endswith(filepath, ".csv")
        filepath *= ".csv"
    end

    # Convert to an absolute path (in case it's not already)
    filepath = abspath(filepath)

    # Check if the file exists before attempting to read it
    if !isfile(filepath)
        error("File '$filepath' does not exist.")
    else
        # Read the CSV file into a DataFrame
        simulation = CSV.read(filepath, DataFrame)
        println("File successfully loaded from: $filepath")
        return simulation
    end
end