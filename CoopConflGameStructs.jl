# some simulation parameters
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

# sadly individual is used by Julia, so I have opted to use "agent" instead
# neatly organizes values for each individual so we don't have to deal with these values spread across different vectors
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

# maybe we should just have functions that requre "pair" to just take 2 "agent"s
# using "pair" may provide clearity, but may also introduce inefficiencies
# idk I think its ok
struct pair
    individual1::agent
    individual2::agent
end

# using dict to keep track of individuals, but this may not be necessary
# all we need is to have a list of individuals, pair them up, calculate their payoffs, and simply create a "new" list of individuals
# so it does not matter to us which individuals are which as long as they interact, reproduce, mutate, etc.
# at least for now
mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, agent}
    pairings::Vector{pair}
    mean_w::Float64
end