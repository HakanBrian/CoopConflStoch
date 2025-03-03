module BehavEqs

export behavioral_equilibrium!, best_response

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population

using ..MainSimulation.Objectives
import ..MainSimulation.Objectives: objective

using Core.Intrinsics

@inline function best_response(
    focal_idx::Int64,
    group::AbstractVector{Int64},
    action_sqrt_view::AbstractVector{Float32},
    action_sqrt_sum::Float32,
    norm_pool::Float32,
    pun_pool::Float32,
    pop::Population,
    delta_action::Float32,
)
    group_size = pop.parameters.group_size
    focal_indiv = @inbounds group[focal_idx]

    # Get the actions
    action_i = @inbounds pop.action[focal_indiv]
    action_i_sqrt = @inbounds action_sqrt_view[focal_idx]
    action_j_filtered_view_sum = action_sqrt_sum - action_i_sqrt

    # Get the internal punishments
    int_pun_ext = @inbounds pop.int_pun_ext[focal_indiv]
    int_pun_self = @inbounds pop.int_pun_self[focal_indiv]

    # Compute norm means
    norm_i = @inbounds pop.norm[focal_indiv]  # Norm of i individual
    norm_mini = (norm_pool * group_size - norm_i) / (group_size - 1)  # Mean norm of -i individuals

    # Calculate current payoff for the individual
    current_payoff = objective(
        action_i,
        action_i_sqrt,
        action_j_filtered_view_sum,
        norm_i,
        norm_mini,
        norm_pool,
        pun_pool,
        int_pun_ext,
        int_pun_self,
    )

    # Perturb action upwards
    action_up = action_i + delta_action
    action_up_sqrt = sqrt_llvm(action_up)

    # Calculate new payoffs with perturbed actions
    new_payoff_up = objective(
        action_up,
        action_up_sqrt,
        action_j_filtered_view_sum,
        norm_i,
        norm_mini,
        norm_pool,
        pun_pool,
        int_pun_ext,
        int_pun_self,
    )

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_up > current_payoff
        return action_up, action_up_sqrt
    end

    # Perturb action downwards
    action_down = max(action_i - delta_action, 0.0f0)
    action_down_sqrt = sqrt_llvm(action_down)

    # Calculate new payoffs with perturbed actions
    new_payoff_down = objective(
        action_down,
        action_down_sqrt,
        action_j_filtered_view_sum,
        norm_i,
        norm_mini,
        norm_pool,
        pun_pool,
        int_pun_ext,
        int_pun_self,
    )

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_down > current_payoff
        return action_down, action_down_sqrt
    end

    return action_i, action_i_sqrt
end

#=
@inline function best_response(
    focal_idx::Int64,
    group::AbstractVector{Int64},
    action_sqrt_view::AbstractVector{Float32},
    action_sqrt_sum::Float32,
    norm_pool::Float32,
    pun_pool::Float32,
    pop::Population,
    delta_action::Float32,
)
    focal_indiv = @inbounds group[focal_idx]

    # Get the actions
    action_i = @inbounds pop.action[focal_indiv]
    action_i_sqrt = @inbounds action_sqrt_view[focal_idx]
    action_j_filtered_view_sum = action_sqrt_sum - action_i_sqrt

    # Get the internal punishments
    int_pun_ext = @inbounds pop.int_pun_ext[focal_indiv]

    # Calculate current payoff for the individual
    current_payoff = objective(
        action_i,
        action_i_sqrt,
        action_j_filtered_view_sum,
        norm_pool,
        pun_pool,
        int_pun_ext,
    )

    # Perturb action upwards
    action_up = action_i + delta_action
    action_up_sqrt = sqrt_llvm(action_up)

    # Calculate new payoffs with perturbed actions
    new_payoff_up = objective(
        action_up,
        action_up_sqrt,
        action_j_filtered_view_sum,
        norm_pool,
        pun_pool,
        int_pun_ext,
    )

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_up > current_payoff
        return action_up, action_up_sqrt
    end

    # Perturb action downwards
    action_down = max(action_i - delta_action, 0.0f0)
    action_down_sqrt = sqrt_llvm(action_down)

    # Calculate new payoffs with perturbed actions
    new_payoff_down = objective(
        action_down,
        action_down_sqrt,
        action_j_filtered_view_sum,
        norm_pool,
        pun_pool,
        int_pun_ext,
    )

    # Decide which direction to adjust action based on payoff improvement
    if new_payoff_down > current_payoff
        return action_down, action_down_sqrt
    end

    return action_i, action_i_sqrt
end
=#

function behavioral_equilibrium!(
    group::AbstractVector{Int64},
    action_sqrt::Vector{Float32},
    action_sqrt_sum::Float32,
    norm_pool::Float32,
    pun_pool::Float32,
    pop::Population,
)
    # Collect parameters
    tolerance = pop.parameters.tolerance
    max_time_steps = pop.parameters.max_time_steps

    # Create a view for group actions
    temp_actions = @inbounds view(pop.action, group)
    action_sqrt_view = @inbounds view(action_sqrt, group)

    # Initialize variables for the loop
    action_change = 1.0f0
    delta_action = 0.1f0
    time_step = 0

    # Iterate until convergence or max time steps
    while time_step < max_time_steps
        time_step += 1

        # Dynamically adjust delta_action
        if delta_action < tolerance
            break
        elseif action_change == 0.0f0
            delta_action *= 0.5f0
        else
            delta_action *= 1.5f0
        end

        action_change = 0.0f0  # Reset action_change for each iteration

        # Calculate the relatively best action of each individual in the group
        for i in eachindex(group)
            best_action, best_action_sqrt = best_response(
                i,
                group,
                action_sqrt_view,
                action_sqrt_sum,
                norm_pool,
                pun_pool,
                pop,
                delta_action,
            )
            diff = abs(best_action - temp_actions[i])
            if diff > action_change
                action_change = diff
            end
            @inbounds temp_actions[i] = best_action
            @inbounds action_sqrt_sum -= action_sqrt_view[i]
            @inbounds action_sqrt_view[i] = best_action_sqrt
            action_sqrt_sum += best_action_sqrt
        end

    end

    nothing
end

end # module BehavioralEquilibriums
