using CSV, FilePathsBase


#########
# Helper ########################################################################################################################
#########

include("structs.jl")


########################
# Population Simulation #########################################################################################################
########################

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
    z_alpha_over_2 = quantile(Normal(), 1 - alpha / 2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * √variance
    upper_bound = z_alpha_over_2 * √variance

    return (lower_bound, upper_bound)
end

##########
# Fitness #######################################################################################################################
##########

function sum_sqrt_loop(actions_j::AbstractVector{Float32})
    sum = 0.0f0
    @inbounds @simd for action_j in actions_j
        sum += sqrt_llvm(action_j)
    end
    return sum  # Return sum of square roots
end

function sqrt_sum_loop(action_i::Float32, actions_j::AbstractVector{Float32})
    sum = 0.0f0
    @inbounds @simd for action_j in actions_j
        sum += action_j
    end
    return sqrt_llvm(action_i + sum)  # Return sqrt of sum
end


#########################
# Behavioral Equilibrium ########################################################################################################
#########################

function filter_out_val!(
    arr::AbstractVector{T},
    exclude_val::T,
    buffer::Vector{T},
) where {T}
    count = 1
    @inbounds for i in eachindex(arr)
        if arr[i] != exclude_val  # Exclude based on the value
            buffer[count] = arr[i]
            count += 1
        end
    end
    return view(buffer, 1:count-1)  # Return a view of the filtered buffer
end

function filter_out_idx!(
    arr::AbstractVector{T},
    exclude_idx::Int,
    buffer::Vector{T},
) where {T}
    count = 1
    @inbounds for i in eachindex(arr)
        if i != exclude_idx  # Exclude based on the index
            buffer[count] = arr[i]
            count += 1
        end
    end
    return view(buffer, 1:count-1)  # Return a view of the filtered buffer
end


#####################
# Social Interaction ############################################################################################################
#####################

function probabilistic_round(x::Float64)::Int64
    lower = floor(Int64, x)
    upper = ceil(Int64, x)
    probability_up = x - lower  # Probability of rounding up

    return rand() < probability_up ? upper : lower
end

function in_place_sample!(data::AbstractVector{T}, k::Int) where {T}
    n = length(data)

    for i in 1:k
        j = rand(i:n)  # Random index between i and n (inclusive)
        data[i], data[j] = data[j], data[i]  # Swap elements
    end
    return @inbounds view(data, 1:k)  # Return a view of the first k elements
end


###############
# Reproduction ##################################################################################################################
###############

function normalize_exponentials(values::Vector{Exponential})
    max_base = maximum(v -> v.base, values)
    sum_probs = 0.0
    probs = similar(values, Float64)  # Pre-allocate for probabilities

    # Compute normalized probabilities
    for (i, v) in pairs(values)
        prob = exp(v.base - max_base)
        probs[i] = prob
        sum_probs += prob
    end

    return probs ./ sum_probs
end


######
# I/O ###########################################################################################################################
######

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

function process_simulation(
    input_dir::String,
    output_dir::String,
    process_function::Function;
    file_extension::String = ".csv",
)
    # Ensure the output directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Get a list of all files in the input directory matching the file extension
    files = filter(f -> endswith(f, file_extension), readdir(input_dir, join = true))

    for file in files
        # Load file
        data = read_simulation(file)

        # Process the data
        processed_data = process_function(data)

        # Create output file path
        output_file = joinpath(output_dir, basename(file))

        # Save processed data
        save_simulation(processed_data, output_file)
    end
end
