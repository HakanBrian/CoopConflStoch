mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    nreps::Int64
    N::Int64
    u::Float64
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
end

mutable struct agent
    id::Int64
    action::Float64
    a::Float64
    p::Float64
    T::Float64
    payoff::Float64
    run_avg_payoff::Float64
    interactions::Int64
end

struct pair
    individual1::agent
    individual2::agent
end

mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, agent}
    pairings::Vector{pair}
    mean_w::Float64
end