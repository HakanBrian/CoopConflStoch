##################
# Simulation Parameters
##################

mutable struct SimulationParameters
    # Game params
    action0::Float32
    norm0::Float32
    ext_pun0::Float32
    int_pun_ext0::Float32
    int_pun_self0::Float32
    # Population-genetic params
    generations::Int64
    max_time_steps::Int64  # behav eq params
    tolerance::Float64  # behav eq params
    population_size::Int64
    group_size::Int64
    synergy::Float32
    relatedness::Float64
    fitness_scaling_factor_a::Float32
    fitness_scaling_factor_b::Float32
    mutation_rate::Float64
    mutation_variance::Float64
    trait_variance::Float64
    # Mutation toggles
    norm_mutation_enabled::Bool
    ext_pun_mutation_enabled::Bool
    int_pun_ext_mutation_enabled::Bool
    int_pun_self_mutation_enabled::Bool
    # File/simulation params
    output_save_tick::Int64  # when to save output
end

function SimulationParameters(;
    action0::Float32=0.5f0,
    norm0::Float32=0.5f0,
    ext_pun0::Float32=0.5f0,
    int_pun_ext0::Float32=0.0f0,
    int_pun_self0::Float32=0.0f0,
    generations::Int64=100000,
    max_time_steps::Int64=100,
    tolerance::Float64=0.01,
    population_size::Int64=50,
    group_size::Int64=10,
    synergy::Float32=0.0f0,
    relatedness::Float64=0.5,
    fitness_scaling_factor_a::Float32=50.0f0,
    fitness_scaling_factor_b::Float32=68.0f0,
    mutation_rate::Float64=0.05,
    mutation_variance::Float64=0.005,
    trait_variance::Float64=0.0,
    norm_mutation_enabled::Bool=true,
    ext_pun_mutation_enabled::Bool=true,
    int_pun_ext_mutation_enabled::Bool=true,
    int_pun_self_mutation_enabled::Bool=true,
    output_save_tick::Int64=10
)
    return SimulationParameters(action0,
                                norm0,
                                ext_pun0,
                                int_pun_ext0,
                                int_pun_self0,
                                generations,
                                max_time_steps,
                                tolerance,
                                population_size,
                                group_size,
                                synergy,
                                relatedness,
                                fitness_scaling_factor_a,
                                fitness_scaling_factor_b,
                                mutation_rate,
                                mutation_variance,
                                trait_variance,
                                norm_mutation_enabled,
                                ext_pun_mutation_enabled,
                                int_pun_ext_mutation_enabled,
                                int_pun_self_mutation_enabled,
                                output_save_tick)
end

function Base.copy(parameters::SimulationParameters)
    return SimulationParameters(
        action0=getfield(parameters, :action0),
        norm0=getfield(parameters, :norm0),
        ext_pun0=getfield(parameters, :ext_pun0),
        int_pun_ext0=getfield(parameters, :int_pun_ext0),
        int_pun_self0=getfield(parameters, :int_pun_self0),
        generations=getfield(parameters, :generations),
        max_time_steps=getfield(parameters, :max_time_steps),
        tolerance=getfield(parameters, :tolerance),
        population_size=getfield(parameters, :population_size),
        group_size=getfield(parameters, :group_size),
        synergy=getfield(parameters, :synergy),
        relatedness=getfield(parameters, :relatedness),
        fitness_scaling_factor_a=getfield(parameters, :fitness_scaling_factor_a),
        fitness_scaling_factor_b=getfield(parameters, :fitness_scaling_factor_b),
        mutation_rate=getfield(parameters, :mutation_rate),
        mutation_variance=getfield(parameters, :mutation_variance),
        trait_variance=getfield(parameters, :trait_variance),
        norm_mutation_enabled=getfield(parameters, :norm_mutation_enabled),
        ext_pun_mutation_enabled=getfield(parameters, :ext_pun_mutation_enabled),
        int_pun_ext_mutation_enabled=getfield(parameters, :int_pun_ext_mutation_enabled),
        int_pun_self_mutation_enabled=getfield(parameters, :int_pun_self_mutation_enabled),
        output_save_tick=getfield(parameters, :output_save_tick)
    )
end

function Base.copy!(old_params::SimulationParameters, new_params::SimulationParameters)
    setfield!(old_params, :action0, getfield(new_params, :action0))
    setfield!(old_params, :norm0, getfield(new_params, :norm0))
    setfield!(old_params, :ext_pun0, getfield(new_params, :ext_pun0))
    setfield!(old_params, :int_pun_ext0, getfield(new_params, :int_pun_ext0))
    setfield!(old_params, :int_pun_self0, getfield(new_params, :int_pun_self0))
    setfield!(old_params, :generations, getfield(new_params, :generations))
    setfield!(old_params, :max_time_steps, getfield(new_params, :max_time_steps))
    setfield!(old_params, :tolerance, getfield(new_params, :tolerance))
    setfield!(old_params, :population_size, getfield(new_params, :population_size))
    setfield!(old_params, :group_size, getfield(new_params, :group_size))
    setfield!(old_params, :synergy, getfield(new_params, :synergy))
    setfield!(old_params, :relatedness, getfield(new_params, :relatedness))
    setfield!(old_params, :fitness_scaling_factor_a, getfield(new_params, :fitness_scaling_factor_a))
    setfield!(old_params, :fitness_scaling_factor_b, getfield(new_params, :fitness_scaling_factor_b))
    setfield!(old_params, :mutation_rate, getfield(new_params, :mutation_rate))
    setfield!(old_params, :mutation_variance, getfield(new_params, :mutation_variance))
    setfield!(old_params, :trait_variance, getfield(new_params, :trait_variance))
    setfield!(old_params, :norm_mutation_enabled, getfield(new_params, :norm_mutation_enabled))
    setfield!(old_params, :ext_pun_mutation_enabled, getfield(new_params, :ext_pun_mutation_enabled))
    setfield!(old_params, :int_pun_ext_mutation_enabled, getfield(new_params, :int_pun_ext_mutation_enabled))
    setfield!(old_params, :int_pun_self_mutation_enabled, getfield(new_params, :int_pun_self_mutation_enabled))
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
    int_pun_ext::Vector{Float32}
    int_pun_self::Vector{Float32}
    payoff::Vector{Float32}
    interactions::Vector{Int64}
end

function Base.copy(pop::Population)
    return Population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :action)),
        copy(getfield(pop, :norm)),
        copy(getfield(pop, :ext_pun)),
        copy(getfield(pop, :int_pun_ext)),
        copy(getfield(pop, :int_pun_self)),
        copy(getfield(pop, :payoff)),
        copy(getfield(pop, :interactions)),
    )
end

function Base.copy!(old_population::Population, new_population::Population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :action), getfield(new_population, :action))
    copy!(getfield(old_population, :norm), getfield(new_population, :norm))
    copy!(getfield(old_population, :ext_pun), getfield(new_population, :ext_pun))
    copy!(getfield(old_population, :int_pun_ext), getfield(new_population, :int_pun_ext))
    copy!(getfield(old_population, :int_pun_self), getfield(new_population, :int_pun_self))
    copy!(getfield(old_population, :payoff), getfield(new_population, :payoff))
    copy!(getfield(old_population, :interactions), getfield(new_population, :interactions))

    nothing
end


##################
# Overloaded Exponential
##################

struct Exponential
    base::Float64  # The exponent value (e.g., 342 for e^342)
end

Base.:-(a::Exponential, b::Exponential) = Exponential(a.base - b.base)  # Overload addition to handle relative weighting (subtracting exponents)