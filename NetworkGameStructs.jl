using StaticArrays

mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    nreps::Int64
    N::Int64
    action::Float64
    a::Float64
    p::Float64
    T::Float64
end

mutable struct population
    parameters::simulation_parameters
    payoffs::Vector{Float64}
    mean_w::Float64
end