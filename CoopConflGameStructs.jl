##################
# Simulation Parameters
##################

mutable struct SimulationParameters
    #game params
    action0::Float32
    norm0::Float32
    ext_pun0::Float32
    int_pun0::Float32
    #popgen params
    gmax::Int  # maximum number of generations
    tmax::Float64  # maximum length of timespan for ODE
    population_size::Int
    synergy::Float32
    relatedness::Float32
    inflation_factor::Int
    fitness_scaling_factor_a::Float32
    fitness_scaling_factor_b::Float32
    mutation_rate::Float32
    mutation_variance::Float32
    trait_variance::Float32
    #file/simulation params
    output_save_tick::Int  # when to save output

    # Constructor with default values
    function SimulationParameters(;
        action0::Float32=0.5f0,
        norm0::Float32=0.5f0,
        ext_pun0::Float32=0.5f0,
        int_pun0::Float32=0.5f0,
        gmax::Int=100000,
        tmax::Float32=5.0f0,
        population_size::Int=50,
        synergy::Float32=0.0f0,
        relatedness::Float32=0.5f0,
        inflation_factor::Int=0,
        fitness_scaling_factor_a::Float32=0.004f0,
        fitness_scaling_factor_b::Float32=10.0f0,
        mutation_rate::Float32=0.05f0,
        mutation_variance::Float32=0.005f0,
        trait_variance::Float32=0.0f0,
        output_save_tick::Int=10
    )
        new(action0, norm0, ext_pun0, int_pun0, gmax, tmax, population_size, synergy, relatedness, inflation_factor, fitness_scaling_factor_a, fitness_scaling_factor_b, mutation_rate, mutation_variance, trait_variance, output_save_tick)
    end
end

function Base.copy(parameters::SimulationParameters)
    return SimulationParameters(
        action0=getfield(parameters, :action0),
        norm0=getfield(parameters, :norm0),
        ext_pun0=getfield(parameters, :ext_pun0),
        int_pun0=getfield(parameters, :int_pun0),
        gmax=getfield(parameters, :gmax),
        tmax=getfield(parameters, :tmax),
        population_size=getfield(parameters, :population_size),
        synergy=getfield(parameters, :synergy),
        relatedness=getfield(parameters, :relatedness),
        inflation_factor=getfield(parameters, :inflation_factor),
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
    setfield!(old_params, :norm0, getfield(new_params, :norm0))
    setfield!(old_params, :ext_pun0, getfield(new_params, :ext_pun0))
    setfield!(old_params, :int_pun0, getfield(new_params, :int_pun0))
    setfield!(old_params, :gmax, getfield(new_params, :gmax))
    setfield!(old_params, :tmax, getfield(new_params, :tmax))
    setfield!(old_params, :population_size, getfield(new_params, :population_size))
    setfield!(old_params, :synergy, getfield(new_params, :synergy))
    setfield!(old_params, :relatedness, getfield(new_params, :relatedness))
    setfield!(old_params, :inflation_factor, getfield(new_params, :inflation_factor))
    setfield!(old_params, :fitness_scaling_factor_a, getfield(new_params, :fitness_scaling_factor_a))
    setfield!(old_params, :fitness_scaling_factor_b, getfield(new_params, :fitness_scaling_factor_b))
    setfield!(old_params, :mutation_rate, getfield(new_params, :mutation_rate))
    setfield!(old_params, :mutation_variance, getfield(new_params, :mutation_variance))
    setfield!(old_params, :trait_variance, getfield(new_params, :trait_variance))
    setfield!(old_params, :output_save_tick, getfield(new_params, :output_save_tick))

    nothing
end


##################
# Population
##################

mutable struct Population
    parameters::SimulationParameters
    action::Vector{Float32}
    norm::Vector{Float32}
    ext_pun::Vector{Float32}
    int_pun::Vector{Float32}
    payoff::Vector{Float32}
    interactions::Vector{Int}
    norm_pool::Float32
    pun_pool::Float32
end

function Base.copy(pop::Population)
    return Population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :action)),
        copy(getfield(pop, :norm)),
        copy(getfield(pop, :ext_pun)),
        copy(getfield(pop, :int_pun)),
        copy(getfield(pop, :payoff)),
        copy(getfield(pop, :interactions)),
        copy(getfield(pop, :norm_pool)),
        copy(getfield(pop, :pun_pool))
    )
end

function Base.copy!(old_population::Population, new_population::Population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :action), getfield(new_population, :action))
    copy!(getfield(old_population, :norm), getfield(new_population, :norm))
    copy!(getfield(old_population, :ext_pun), getfield(new_population, :ext_pun))
    copy!(getfield(old_population, :int_pun), getfield(new_population, :int_pun))
    copy!(getfield(old_population, :payoff), getfield(new_population, :payoff))
    copy!(getfield(old_population, :interactions), getfield(new_population, :interactions))
    copy!(getfield(old_population, :norm_pool), getfield(new_population, :norm_pool))
    copy!(getfield(old_population, :pun_pool), getfield(new_population, :pun_pool))

    nothing
end