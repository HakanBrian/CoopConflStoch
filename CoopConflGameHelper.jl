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

function sweep_statistics(all_simulation_means::DataFrame, r_values::Vector{Float64})
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

    # Initialize an empty DataFrame to store last rows
    last_rows = DataFrame()

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Append the last row of `statistics` to `last_rows`
        push!(last_rows, statistics[end, :])
    end

    rename!(last_rows, :generation => :relatedness)

    last_rows.relatedness = r_values

    return last_rows
end

function sweep_statistics(all_simulation_means::DataFrame, r_values::Vector{Float64}, ep_values::Vector{Float32})
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

    # Initialize an empty DataFrame to store last rows
    last_rows = DataFrame()

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Append the last row of `statistics` to `last_rows`
        push!(last_rows, statistics[end, :])
    end

    rename!(last_rows, :generation => :relatedness)
    last_rows.relatedness = repeat(r_values, inner = length(ep_values))

    insertcols!(last_rows, 2, :ext_pun => repeat(ep_values, length(r_values)))
    select!(last_rows, Not([:p_mean_mean, :p_mean_std]))

    return last_rows
end

function sweep_statistics(all_simulation_means::DataFrame, r_values::Vector{Float64}, gs_values::Vector{Int64})
    # Determine the number of params
    num_params = maximum(all_simulation_means.param_id)

    # Initialize an empty DataFrame to store last rows
    last_rows = DataFrame()

    for i in 1:num_params
        # Filter rows by `param_id`
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Append the last row of `statistics` to `last_rows`
        push!(last_rows, statistics[end, :])
    end

    rename!(last_rows, :generation => :relatedness)
    last_rows.relatedness = repeat(r_values, inner = length(gs_values))

    insertcols!(last_rows, 2, :group_size => repeat(gs_values, length(r_values)))

    return last_rows
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