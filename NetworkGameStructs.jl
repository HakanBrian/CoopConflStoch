using StaticArrays

mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    nreps::Int64
    N::Int64
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
end

mutable struct network
    genotype_id::Int64
    Wm::SMatrix
    Wb::SVector
    InitialOffer::Float64
    CurrentOffer::Float64
end

mutable struct population
    parameters::simulation_parameters
    shuffled_indices::Vector{Int64}
    action::Vector{Float64}
    a::Vector{Float64}
    p::Vector{Float64}
    T::Vector{Float64}
    payoffs::Vector{Float64}
    mean_w::Float64
end

mutable struct pop_parameters
    Tuples::Vector{Tuple{Float64,Float64,Float64,Float64}}
end