##################
# Simulation Parameters
##################

mutable struct SimulationParameters
    #game params
    action0::Float64
    a0::Float64
    p0::Float64
    T0::Float64
    #popgen params
    gmax::Int64  # maximum number of generations
    tmax::Float64  # maximum length of timespan for ODE
    population_size::Int64
    group_size::Int64
    synergy::Float64
    relatedness::Float64
    fitness_scaling_factor_a::Float64
    fitness_scaling_factor_b::Float64
    mutation_rate::Float64
    mutation_variance::Float64
    trait_variance::Float64
    #file/simulation params
    output_save_tick::Int64  # when to save output

    # Constructor with default values
    function SimulationParameters(;
        action0::Float64=0.5,
        a0::Float64=0.5,
        p0::Float64=0.5,
        T0::Float64=0.0,
        gmax::Int64=100000,
        tmax::Float64=5.0,
        population_size::Int64=50,
        group_size::Int64 = 10,
        synergy::Float64=0.0,
        relatedness::Float64=0.5,
        fitness_scaling_factor_a::Float64=0.004,
        fitness_scaling_factor_b::Float64=10.0,
        mutation_rate::Float64=0.05,
        mutation_variance::Float64=0.005,
        trait_variance::Float64=0.0,
        output_save_tick::Int64=10
    )
        new(action0, a0, p0, T0, gmax, tmax, population_size, group_size, synergy, relatedness, fitness_scaling_factor_a, fitness_scaling_factor_b, mutation_rate, mutation_variance, trait_variance, output_save_tick)
    end
end

function Base.copy(parameters::SimulationParameters)
    return SimulationParameters(
        action0=getfield(parameters, :action0),
        a0=getfield(parameters, :a0),
        p0=getfield(parameters, :p0),
        T0=getfield(parameters, :T0),
        gmax=getfield(parameters, :gmax),
        tmax=getfield(parameters, :tmax),
        population_size=getfield(parameters, :population_size),
        group_size=getfield(parameters, :group_size),
        synergy=getfield(parameters, :synergy),
        relatedness=getfield(parameters, :relatedness),
        fitness_scaling_factor_a=getfield(parameters, :fitness_scaling_factor_a),
        fitness_scaling_factor_b=getfield(parameters, :fitness_scaling_factor_b),
        mutation_rate=getfield(parameters, :mutation_rate),
        mutation_variance=getfield(parameters, :mutation_variance),
        trait_variance=getfield(parameters, :trait_variance),
        output_save_tick=getfield(parameters, :output_save_tick)
    )
end

function Base.copy!(old_params::SimulationParameters, new_params::SimulationParameters)
    setfield!(old_params, :action0, getfield(new_params, :action0))
    setfield!(old_params, :a0, getfield(new_params, :a0))
    setfield!(old_params, :p0, getfield(new_params, :p0))
    setfield!(old_params, :T0, getfield(new_params, :T0))
    setfield!(old_params, :gmax, getfield(new_params, :gmax))
    setfield!(old_params, :tmax, getfield(new_params, :tmax))
    setfield!(old_params, :population_size, getfield(new_params, :population_size))
    setfield!(old_params, :group_size, getfield(new_params, :group_size))
    setfield!(old_params, :synergy, getfield(new_params, :synergy))
    setfield!(old_params, :relatedness, getfield(new_params, :relatedness))
    setfield!(old_params, :fitness_scaling_factor_a, getfield(new_params, :fitness_scaling_factor_a))
    setfield!(old_params, :fitness_scaling_factor_b, getfield(new_params, :fitness_scaling_factor_b))
    setfield!(old_params, :mutation_rate, getfield(new_params, :mutation_rate))
    setfield!(old_params, :mutation_variance, getfield(new_params, :mutation_variance))
    setfield!(old_params, :trait_variance, getfield(new_params, :trait_variance))
    setfield!(old_params, :output_save_tick, getfield(new_params, :output_save_tick))

    nothing
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
    interactions::Int64
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

    nothing
end


##################
# Population
##################

mutable struct Population
    parameters::SimulationParameters
    individuals::Dict{Int64, Individual}
    norm_pool::Float64
    punishment_pool::Float64
end

function Base.copy(pop::Population)
    return Population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :individuals)),
        copy(getfield(pop, :norm_pool)),
        copy(getfield(pop, :punishment_pool))
    )
end

function Base.copy!(old_population::Population, new_population::Population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :individuals), getfield(new_population, :individuals))
    copy!(getfield(old_population, :norm_pool), getfield(new_population, :norm_pool))
    copy!(getfield(old_population, :punishment_pool), getfield(new_population, :punishment_pool))

    nothing
end