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
    gmax::Int64  # maximum number of generations
    tmax::Float32  # maximum length of timespan for ODE
    population_size::Int64
    synergy::Float64
    relatedness::Float64
    fitness_scaling_factor::Float64
    mutation_rate::Float64
    trait_variance::Float64
    mutation_variance::Float64
    #file/simulation params
    output_save_tick::Int64  # when to save output
end

function Base.copy(parameters::simulation_parameters)
    return simulation_parameters(
        getfield(parameters, :action0),
        getfield(parameters, :a0),
        getfield(parameters, :p0),
        getfield(parameters, :T0),
        getfield(parameters, :gmax),
        getfield(parameters, :tmax),
        getfield(parameters, :population_size),
        getfield(parameters, :synergy),
        getfield(parameters, :relatedness),
        getfield(parameters, :fitness_scaling_factor),
        getfield(parameters, :mutation_rate),
        getfield(parameters, :trait_variance),
        getfield(parameters, :mutation_variance),
        getfield(parameters, :output_save_tick)
    )
end

function Base.copy!(old_params::simulation_parameters, new_params::simulation_parameters)
    setfield!(old_params, :action0, getfield(new_params, :action0))
    setfield!(old_params, :a0, getfield(new_params, :a0))
    setfield!(old_params, :p0, getfield(new_params, :p0))
    setfield!(old_params, :T0, getfield(new_params, :T0))
    setfield!(old_params, :gmax, getfield(new_params, :gmax))
    setfield!(old_params, :tmax, getfield(new_params, :tmax))
    setfield!(old_params, :population_size, getfield(new_params, :population_size))
    setfield!(old_params, :synergy, getfield(new_params, :synergy))
    setfield!(old_params, :relatedness, getfield(new_params, :relatedness))
    setfield!(old_params, :fitness_scaling_factor, getfield(new_params, :fitness_scaling_factor))
    setfield!(old_params, :mutation_rate, getfield(new_params, :mutation_rate))
    setfield!(old_params, :trait_variance, getfield(new_params, :trait_variance))
    setfield!(old_params, :mutation_variance, getfield(new_params, :mutation_variance))
    setfield!(old_params, :output_save_tick, getfield(new_params, :output_save_tick))

    nothing
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
    return individual(
        getfield(ind, :action),
        getfield(ind, :a),
        getfield(ind, :p),
        getfield(ind, :T),
        getfield(ind, :payoff),
        getfield(ind, :interactions)
    )
end

function Base.copy!(old_ind::individual, new_ind::individual)
    setfield!(old_ind, :action, getfield(new_ind, :action))
    setfield!(old_ind, :a, getfield(new_ind, :a))
    setfield!(old_ind, :p, getfield(new_ind, :p))
    setfield!(old_ind, :T, getfield(new_ind, :T))
    setfield!(old_ind, :payoff, getfield(new_ind, :payoff))
    setfield!(old_ind, :interactions, getfield(new_ind, :interactions))

    nothing
end


##################
# Population
##################

mutable struct population
    parameters::simulation_parameters
    individuals::Dict{Int64, individual}
    norm_pool::Float64
    punishment_pool::Float64
end

function Base.copy(pop::population)
    return population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :individuals)),
        copy(getfield(pop, :norm_pool)),
        copy(getfield(pop, :punishment_pool))
    )
end

function Base.copy!(old_population::population, new_population::population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :individuals), getfield(new_population, :individuals))
    copy!(getfield(old_population, :norm_pool), getfield(new_population, :norm_pool))
    copy!(getfield(old_population, :punishment_pool), getfield(new_population, :punishment_pool))

    nothing
end