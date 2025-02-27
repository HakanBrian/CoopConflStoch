module SocialInteractions 

export social_interactions!, total_payoff!, shuffle_and_group

using ..Populations, ..Objectives, ..BehavEqs, Core.Intrinsics, Random

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

function shuffle_and_group(
    groups::Matrix{Int64},
    population_size::Int64,
    group_size::Int64,
    relatedness::Float64,
)
    # Collect and shuffle individual indices
    individuals_indices = collect(1:population_size)
    shuffle!(individuals_indices)

    # Pre-allocate a buffer for candidates (one less than population size, since focal individual is excluded)
    candidates_buffer = Vector{Int64}(undef, population_size - 1)

    # Iterate over each individual index and form a group
    for i in 1:population_size
        focal_individual_index = individuals_indices[i]

        # Filter out the focal individual
        candidates_filtered_view =
            filter_out_val!(individuals_indices, focal_individual_index, candidates_buffer)

        # Calculate the number of related and random individuals
        num_related = probabilistic_round(relatedness * (group_size - 1))
        num_random = group_size - num_related - 1

        # Assign the focal individual to the group
        groups[i, :] .= focal_individual_index

        # Assign random individuals to the group
        groups[i, end-num_random+1:end] =
            in_place_sample!(candidates_filtered_view, num_random)
    end

    return groups
end

function total_payoff!(
    group::AbstractVector{Int64},
    norm_pool::Float32,
    pun_pool::Float32,
    pop::Population,
)
    focal_idiv = group[1]

    # Extract the action of the focal individual as a real number (not a view)
    action_i = @inbounds pop.action[focal_idiv]

    # Collect actions from the other individuals in the group
    actions_j = @inbounds @view pop.action[@view group[2:end]]

    # Compute the payoff for the focal individual
    payoff_foc = payoff(action_i, actions_j, norm_pool, pun_pool)

    # Update the individual's payoff and interactions
    pop.payoff[focal_idiv] =
        (payoff_foc + pop.interactions[focal_idiv] * pop.payoff[focal_idiv]) /
        (pop.interactions[focal_idiv] + 1)
    pop.interactions[focal_idiv] += 1

    nothing
end

function find_actions_payoffs!(
    final_actions::Vector{Float32},
    action_sqrt::Vector{Float32},
    groups::Matrix{Int64},
    pop::Population,
)
    # Iterate over each group to find actions, sqrt of actions, and payoffs
    for i in axes(groups, 1)
        group = @inbounds @view groups[i, :]

        norm_pool = 0.0f0
        pun_pool = 0.0f0
        action_sqrt_sum = 0.0f0
        for member in group
            @inbounds norm_pool += pop.norm[member]
            @inbounds pun_pool += pop.ext_pun[member]
            @inbounds action_sqrt_sum += action_sqrt[member]
        end
        norm_pool /= pop.parameters.group_size
        pun_pool /= pop.parameters.group_size

        # Calculate equilibrium actions then payoffs for current groups
        behavioral_equilibrium!(
            group,
            action_sqrt,
            action_sqrt_sum,
            norm_pool,
            pun_pool,
            pop,
        )
        total_payoff!(group, norm_pool, pun_pool, pop)

        # Update final action of the focal individual
        final_actions[group[1]] = pop.action[group[1]]
    end
end

function social_interactions!(pop::Population)
    # Pre-allocate vectors
    final_actions = Vector{Float32}(undef, pop.parameters.population_size)
    action_sqrt = Vector{Float32}(undef, pop.parameters.population_size)

    # Cache the square root of actions
    action_sqrt = map(action -> sqrt_llvm(action), pop.action)

    # Shuffle and group individuals
    groups = shuffle_and_group(
        pop.groups,
        pop.parameters.population_size,
        pop.parameters.group_size,
        pop.parameters.relatedness,
    )

    # Get actions while updating payoff
    find_actions_payoffs!(final_actions, action_sqrt, groups, pop)

    # Update the population values with the equilibrium actions
    pop.action = final_actions

    nothing
end

end # module SocialInteractions
