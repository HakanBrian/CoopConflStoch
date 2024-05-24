##################
# simulation parameters
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
    u::Float64  # mutation rate
    trait_var::Float64  # trait variance
    mut_Var::Float64  # mutation variance
    #file/simulation params
    output_save_tick::Int64
end

function Base.copy(parameters::simulation_parameters)
    field_values = map(field -> getfield(parameters, field), fieldnames(simulation_parameters))
    return simulation_parameters(field_values...)
end

function Base.copy!(old_params::simulation_parameters, new_params::simulation_parameters)
    field_names = fieldnames(simulation_parameters)
    for name in field_names
        setfield!(old_params, name, getfield(new_params, name))
    end
end


##################
# individual
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
    field_values = map(field -> getfield(ind, field), fieldnames(individual))
    return individual(field_values...)
end

function Base.copy!(old_ind::individual, new_ind::individual)
    field_names = fieldnames(individual)
    for name in field_names
        setfield!(old_ind, name, getfield(new_ind, name))
    end
end

function Base.copy(inds::Dict{Int64, individual})
    return Dict{Int64, individual}(key => copy(value) for (key, value) in inds)
end

function Base.copy!(old_inds::Dict{Int64, individual}, new_inds::Dict{Int64, individual})
    for key in keys(old_inds)
        copy!(old_inds[key], new_inds[key])
    end
end


##################
# population
##################

mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, individual}
    old_individuals::Dict{Int64, individual}
end

function Base.copy(pop::population)
    field_values = map(field -> copy(getfield(pop, field)), fieldnames(population))
    return population(field_values...)
end

function Base.copy!(old_population::population, new_population::population)
    for field in fieldnames(population)
        copy!(getfield(old_population, field), getfield(new_population, field))
    end
end