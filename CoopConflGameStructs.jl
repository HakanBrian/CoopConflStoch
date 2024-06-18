##################
# Simulation Parameters
##################

mutable struct Simulation_Parameters
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
    #popgen params
    gmax::Int64  # max generations
    tmax::Float32  # max timespan for ODE
    N::Int64  # population size
    v::Float64  # synergy
    u::Float64  # mutation rate
    trait_var::Float64  # trait variance
    mut_var::Float64  # mutation variance
    #file/simulation params
    output_save_tick::Int64
end

function Base.copy(parameters::Simulation_Parameters)
    return Simulation_Parameters(
        getfield(parameters, :action0),
        getfield(parameters, :a0),
        getfield(parameters, :p0),
        getfield(parameters, :T0),
        getfield(parameters, :gmax),
        getfield(parameters, :tmax),
        getfield(parameters, :N),
        getfield(parameters, :v),
        getfield(parameters, :u),
        getfield(parameters, :trait_var),
        getfield(parameters, :mut_var),
        getfield(parameters, :output_save_tick)
    )
end

function Base.copy!(old_params::Simulation_Parameters, new_params::Simulation_Parameters)
    setfield!(old_params, :action0, getfield(new_params, :action0))
    setfield!(old_params, :a0, getfield(new_params, :a0))
    setfield!(old_params, :p0, getfield(new_params, :p0))
    setfield!(old_params, :T0, getfield(new_params, :T0))
    setfield!(old_params, :gmax, getfield(new_params, :gmax))
    setfield!(old_params, :tmax, getfield(new_params, :tmax))
    setfield!(old_params, :N, getfield(new_params, :N))
    setfield!(old_params, :v, getfield(new_params, :v))
    setfield!(old_params, :u, getfield(new_params, :u))
    setfield!(old_params, :trait_var, getfield(new_params, :trait_var))
    setfield!(old_params, :mut_var, getfield(new_params, :mut_var))
    setfield!(old_params, :output_save_tick, getfield(new_params, :output_save_tick))

    return nothing
end


##################
# Individual
##################

mutable struct Individual
    action::Float64
    a::Float64
    p::Float64
    T::Float64
    payoff::Float64
    interaction::Int64
end

function Base.copy(ind::Individual)
    return Individual(
        getfield(ind, :action),
        getfield(ind, :a),
        getfield(ind, :p),
        getfield(ind, :T),
        getfield(ind, :payoff),
        getfield(ind, :interactions)
    )
end

function Base.copy!(old_ind::Individual, new_ind::Individual)
    setfield!(old_ind, :action, getfield(new_ind, :action))
    setfield!(old_ind, :a, getfield(new_ind, :a))
    setfield!(old_ind, :p, getfield(new_ind, :p))
    setfield!(old_ind, :T, getfield(new_ind, :T))
    setfield!(old_ind, :payoff, getfield(new_ind, :payoff))
    setfield!(old_ind, :interactions, getfield(new_ind, :interactions))

    return nothing
end


##################
# Population
##################

mutable struct Population
    parameters::Simulation_Parameters
    actions::Vector{Float64}
    as::Vector{Float64}
    ps::Vector{Float64}
    Ts::Vector{Float64}
    payoffs::Vector{Float64}
    interactions::Vector{Int64}
    norm_pool::Float64
    punishment_pool::Float64
end

function get_individual(population::Population, i::Int64)
    Individual(population.actions[i], population.as[i], population.ps[i], population.Ts[i], population.payoffs[i], population.interactions[i])
end

function set_individual!(population::Population, i::Int64, individual::Individual)
    population.actions[i] = individual.action
    population.as[i] = individual.a
    population.ps[i] = individual.p
    population.Ts[i] = individual.T
    population.payoffs[i] = individual.payoff
    population.interactions[i] = individual.interaction

    nothing
end

function Base.copy(pop::Population)
    return Population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :actions)),
        copy(getfield(pop, :as)),
        copy(getfield(pop, :ps)),
        copy(getfield(pop, :Ts)),
        copy(getfield(pop, :payoffs)),
        copy(getfield(pop, :interactions)),
        copy(getfield(pop, :norm_pool)),
        copy(getfield(pop, :punishment_pool))
    )
end

function Base.copy!(old_population::Population, new_population::Population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :actions), getfield(new_population, :actions))
    copy!(getfield(old_population, :a), getfield(new_population, :as))
    copy!(getfield(old_population, :p), getfield(new_population, :ps))
    copy!(getfield(old_population, :T), getfield(new_population, :Ts))
    copy!(getfield(old_population, :payoff), getfield(new_population, :payoffs))
    copy!(getfield(old_population, :interactions), getfield(new_population, :interactions))
    copy!(getfield(old_population, :norm_pool), getfield(new_population, :norm_pool))
    copy!(getfield(old_population, :punishment_pool), getfield(new_population, :punishment_pool))

    return nothing
end