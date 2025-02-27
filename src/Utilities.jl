module Utilities

export benefit
export benefit_sqrt
export cost
export external_punishment
export internal_punishment_I
export internal_punishment_II
export internal_punishment_ext
export internal_punishment_self

using Core.Intrinsics


##########
# Benefit #######################################################################################################################
##########

@inline function benefit(action_i::Float32, actions_j::AbstractVector{Float32})
    sqrt_action_i = sqrt_llvm(action_i)
    sum_sqrt_actions_j = sum_sqrt_loop(actions_j)

    return sqrt_action_i + sum_sqrt_actions_j
end

@inline function benefit(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    synergy::Float32,
)
    sqrt_action_i = sqrt_llvm(action_i)
    sum_sqrt_actions_j = sum_sqrt_loop(actions_j)
    sum_sqrt_actions = sqrt_action_i + sum_sqrt_actions_j
    sqrt_sum_actions = sqrt_sum_loop(action_i, ctions_j)

    return (1 - synergy) * sum_sqrt_actions + synergy * sqrt_sum_actions
end

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

@inline function benefit_sqrt(action_i::Float32, actions_j::Float32)
    return action_i + actions_j
end


#######
# Cost ##########################################################################################################################
#######

@inline function cost(action_i::Float32)
    return action_i^2
end


######################
# External Punishemnt ###########################################################################################################
######################

@inline function external_punishment(
    action_i::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
)
    return punishment_pool * (action_i - norm_pool)^2
end


######################
# Internal Punishemnt ###########################################################################################################
######################

@inline function internal_punishment_I(
    action_i::Float32,
    norm_pool::Float32,
    T_ext::Float32,
)
    return T_ext * (action_i - norm_pool)^2
end

@inline function internal_punishment_II(
    action_i::Float32,
    norm_pool::Float32,
    T_ext::Float32,
)
    return T_ext * log(1 + ((action_i - norm_pool)^2))
end

@inline function internal_punishment_ext(
    action_i::Float32,
    norm_pool_mini::Float32,
    T_ext::Float32,
)
    return T_ext * (action_i - norm_pool_mini)^2
end

@inline function internal_punishment_self(
    action_i::Float32,
    norm_i::Float32,
    T_self::Float32,
)
    return T_self * (action_i - norm_i)^2
end

@inline function benefit(action_i::Float32, actions_j::AbstractVector{Float32})
    sqrt_action_i = sqrt_llvm(action_i)
    sum_sqrt_actions_j = sum_sqrt_loop(actions_j)

    return sqrt_action_i + sum_sqrt_actions_j
end

end # module Utilities
