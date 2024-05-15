# some simulation parameters
mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    N::Int64  # population size
    u::Float64  # mutation rate
    var::Float64  # trait variance
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

function Base.copy!(old_ind::individual, new_ind::individual)
    old_ind.action = new_ind.action
    old_ind.a = new_ind.a
    old_ind.p = new_ind.p
    old_ind.T = new_ind.T
end

function Base.copy(inds::Dict{Int64, individual})
    return Dict{Int64, individual}(key => copy(value) for (key, value) in inds)
end

function Base.copy!(old_inds::Dict{Int64, individual}, new_inds::Dict{Int64, individual})
    for key in keys(old_inds)
        copy!(old_inds[key], new_inds[key])
    end
end

# Add old individuals
mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, individual}
    old_individuals::Dict{Int64, individual}
end