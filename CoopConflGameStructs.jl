# some simulation parameters
mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    N::Int64
    u::Float64
    var::Float64
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
end

# organizes values for each individual
mutable struct individual
    action::Float64
    a::Float64
    p::Float64
    T::Float64
    payoff::Float64
    interactions::Int64
end

function Base.copy(ind::individual)
    return individual(ind.action, ind.a, ind.p, ind.T, ind.payoff, ind.interactions)
end

function Base.copy(inds::Dict{Int64, individual})
    return Dict{Int64, individual}(key => copy(value) for (key, value) in inds)
end

mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, individual}
    mean_w::Float64
end