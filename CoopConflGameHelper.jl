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