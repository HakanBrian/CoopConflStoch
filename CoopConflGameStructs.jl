##################
# Simulation Parameters
##################

mutable struct simulation_parameters
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
    #popgen params
    gmax::Int64  # max generations
    tmax::Int64  # max timespan for ODE
    N::Int64  # population size
    v::Float64  # synergy
    u::Float64  # mutation rate
    trait_var::Float64  # trait variance
    mut_var::Float64  # mutation variance
    #file/simulation params
    output_save_tick::Int64
end

function Base.copy(parameters::simulation_parameters)
    return simulation_parameters(getfield(parameters, :action0), getfield(parameters, :a0), getfield(parameters, :p0), getfield(parameters, :T0), 
                                 getfield(parameters, :gmax), getfield(parameters, :tmax), getfield(parameters, :N), getfield(parameters, :v), 
                                 getfield(parameters, :u), getfield(parameters, :trait_var), getfield(parameters, :mut_var), getfield(parameters, :output_save_tick))
end

function Base.copy!(old_params::simulation_parameters, new_params::simulation_parameters)
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

mutable struct individual
    action::Float64
    a::Float64
    p::Float64
    T::Float64
    payoff::Float64
    interactions::Int64
end

function Base.copy(ind::individual)
    return individual(getfield(ind, :action), getfield(ind, :a), getfield(ind, :p), getfield(ind, :T), getfield(ind, :payoff), getfield(ind, :interactions))
end

function Base.copy!(old_ind::individual, new_ind::individual)
    setfield!(old_ind, :action, getfield(new_ind, :action))
    setfield!(old_ind, :a, getfield(new_ind, :a))
    setfield!(old_ind, :p, getfield(new_ind, :p))
    setfield!(old_ind, :T, getfield(new_ind, :T))
    setfield!(old_ind, :payoff, getfield(new_ind, :payoff))
    setfield!(old_ind, :interactions, getfield(new_ind, :interactions))

    return nothing
end

function Base.copy(inds::Dict{Int64, individual})
    return Dict{Int64, individual}(key => copy(value) for (key, value) in inds)
end

function Base.copy!(old_inds::Dict{Int64, individual}, new_inds::Dict{Int64, individual})
    for key in keys(old_inds)
        new_value = new_inds[key]
        copy!(old_inds[key], new_value)
    end

    return nothing
end


##################
# Population
##################

mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, individual}
    old_individuals::Dict{Int64, individual}
end

function Base.copy(pop::population)
    return population(getfield(pop, :parameters), getfield(pop, :individuals), getfield(pop, :old_individuals))
end

function Base.copy!(old_population::population, new_population::population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :individuals), getfield(new_population, :individuals))
    copy!(getfield(old_population, :old_individuals), getfield(new_population, :old_individuals))

    return nothing
end