module Objectives

export objective, payoff, fitness, fitness_exp, fitness_exp_norm

using ..Utilities

# Normal version =================================
@inline function payoff(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_pool::Float32,
    punishment_pool::Float32,
)
    b = benefit(action_i, actions_j)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
)::Float32
    p = payoff(action_i, actions_j, norm_pool, punishment_pool)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_i::Float32,
    norm_mini::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
    T_self::Float32,
)::Float32
    p = payoff(action_i, actions_j, norm_pool, punishment_pool)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

# Normal Sqrt version =================================
@inline function payoff(
    action_i::Float32,
    action_i_sqrt::Float32,
    actions_j::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
)
    b = benefit_sqrt(action_i_sqrt, actions_j)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(
    action_i::Float32,
    action_i_sqrt::Float32,
    actions_j::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
)::Float32
    p = payoff(action_i, action_i_sqrt, actions_j, norm_pool, punishment_pool)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(
    action_i::Float32,
    action_i_sqrt::Float32,
    actions_j::Float32,
    norm_i::Float32,
    norm_mini::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
    T_self::Float32,
)::Float32
    p = payoff(action_i, action_i_sqrt, actions_j, norm_pool, punishment_pool)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

# Synergy version =================================
@inline function payoff(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_pool::Float32,
    punishment_pool::Float32,
    synergy::Float32,
)
    b = benefit(action_i, actions_j, synergy)
    c = cost(action_i)
    ep = external_punishment(action_i, norm_pool, punishment_pool)
    return b - c - ep
end

@inline function objective(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
    synergy::Float32,
)::Float32
    p = payoff(action_i, actions_j, norm_pool, punishment_pool, synergy)
    i = internal_punishment_I(action_i, norm_pool, T_ext)
    return p - i
end

@inline function objective(
    action_i::Float32,
    actions_j::AbstractVector{Float32},
    norm_i::Float32,
    norm_mini::Float32,
    norm_pool::Float32,
    punishment_pool::Float32,
    T_ext::Float32,
    T_self::Float32,
    synergy::Float32,
)::Float32
    p = payoff(action_i, actions_j, norm_pool, punishment_pool, synergy)
    ipe = internal_punishment_ext(action_i, norm_mini, T_ext)
    ips = internal_punishment_self(action_i, norm_i, T_self)
    return p - ipe - ips
end

end # module Objective
