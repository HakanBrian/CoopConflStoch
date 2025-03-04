module SimulationParameters

export SimulationParameter, update_params, generate_params

mutable struct SimulationParameter
    # Game parameters
    action0::Float32
    norm0::Float32
    ext_pun0::Float32
    int_pun_ext0::Float32
    int_pun_self0::Float32
    # Population-genetic parameters
    generations::Int64
    max_time_steps::Int64  # Behavioral equilibrium params
    tolerance::Float64
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
    # Function toggles
    use_bipenal::Bool
    # File/simulation parameters
    output_save_tick::Int64
end

function SimulationParameter(;
    action0::Float32 = 0.5f0,
    norm0::Float32 = 0.5f0,
    ext_pun0::Float32 = 0.5f0,
    int_pun_ext0::Float32 = 0.0f0,
    int_pun_self0::Float32 = 0.0f0,
    generations::Int64 = 100000,
    max_time_steps::Int64 = 100,
    tolerance::Float64 = 0.01,
    population_size::Int64 = 50,
    group_size::Int64 = 10,
    synergy::Float32 = 0.0f0,
    relatedness::Float64 = 0.5,
    fitness_scaling_factor_a::Float32 = 50.0f0,
    fitness_scaling_factor_b::Float32 = 68.0f0,
    mutation_rate::Float64 = 0.05,
    mutation_variance::Float64 = 0.005,
    trait_variance::Float64 = 0.0,
    norm_mutation_enabled::Bool = true,
    ext_pun_mutation_enabled::Bool = true,
    int_pun_ext_mutation_enabled::Bool = true,
    int_pun_self_mutation_enabled::Bool = true,
    use_bipenal::Bool = true,
    output_save_tick::Int64 = 10,
)
    return SimulationParameter(
        action0,
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
        use_bipenal,
        output_save_tick,
    )
end

function update_params(base_params::SimulationParameter; kwargs...)
    # Update parameters by merging base parameters with new parameters
    return SimulationParameter(;
        merge(
            Dict(
                fieldname => getfield(base_params, fieldname) for
                fieldname in fieldnames(SimulationParameter)
            ),
            kwargs,
        )...,
    )
end

function generate_params(
    base_params::SimulationParameter,
    sweep_vars::Dict{Symbol,<:AbstractVector},
    linked_params = Dict{Symbol,Symbol}(),
)
    # Filter out linked parameters whose source is NOT in `sweep_vars`
    valid_linked_parameters =
        Dict(k => v for (k, v) in linked_params if v in keys(sweep_vars))

    # Sort primary keys alphabetically (excluding linked parameters)
    primary_keys = sort(collect(setdiff(keys(sweep_vars), keys(valid_linked_parameters))))

    # Generate parameter list with ordered combinations
    parameters = vec([
        update_params(
            base_params;
            NamedTuple{Tuple(primary_keys)}(values)...,
            Dict(
                k => values[findfirst(==(v), primary_keys)] for
                (k, v) in valid_linked_parameters if
                findfirst(==(v), primary_keys) !== nothing
            )...,
        ) for values in Iterators.product((sweep_vars[k] for k in primary_keys)...)
    ])

    return parameters
end

function Base.copy(parameters::SimulationParameter)
    return SimulationParameters(
        action0 = getfield(parameters, :action0),
        norm0 = getfield(parameters, :norm0),
        ext_pun0 = getfield(parameters, :ext_pun0),
        int_pun_ext0 = getfield(parameters, :int_pun_ext0),
        int_pun_self0 = getfield(parameters, :int_pun_self0),
        generations = getfield(parameters, :generations),
        max_time_steps = getfield(parameters, :max_time_steps),
        tolerance = getfield(parameters, :tolerance),
        population_size = getfield(parameters, :population_size),
        group_size = getfield(parameters, :group_size),
        synergy = getfield(parameters, :synergy),
        relatedness = getfield(parameters, :relatedness),
        fitness_scaling_factor_a = getfield(parameters, :fitness_scaling_factor_a),
        fitness_scaling_factor_b = getfield(parameters, :fitness_scaling_factor_b),
        mutation_rate = getfield(parameters, :mutation_rate),
        mutation_variance = getfield(parameters, :mutation_variance),
        trait_variance = getfield(parameters, :trait_variance),
        norm_mutation_enabled = getfield(parameters, :norm_mutation_enabled),
        ext_pun_mutation_enabled = getfield(parameters, :ext_pun_mutation_enabled),
        int_pun_ext_mutation_enabled = getfield(parameters, :int_pun_ext_mutation_enabled),
        int_pun_self_mutation_enabled = getfield(
            parameters,
            :int_pun_self_mutation_enabled,
        ),
        use_bipenal = getfield(parameters, :use_bipenal),
        output_save_tick = getfield(parameters, :output_save_tick),
    )
end

function Base.copy!(old_params::SimulationParameter, new_params::SimulationParameter)
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
    setfield!(
        old_params,
        :fitness_scaling_factor_a,
        getfield(new_params, :fitness_scaling_factor_a),
    )
    setfield!(
        old_params,
        :fitness_scaling_factor_b,
        getfield(new_params, :fitness_scaling_factor_b),
    )
    setfield!(old_params, :mutation_rate, getfield(new_params, :mutation_rate))
    setfield!(old_params, :mutation_variance, getfield(new_params, :mutation_variance))
    setfield!(old_params, :trait_variance, getfield(new_params, :trait_variance))
    setfield!(
        old_params,
        :norm_mutation_enabled,
        getfield(new_params, :norm_mutation_enabled),
    )
    setfield!(
        old_params,
        :ext_pun_mutation_enabled,
        getfield(new_params, :ext_pun_mutation_enabled),
    )
    setfield!(
        old_params,
        :int_pun_ext_mutation_enabled,
        getfield(new_params, :int_pun_ext_mutation_enabled),
    )
    setfield!(
        old_params,
        :int_pun_self_mutation_enabled,
        getfield(new_params, :int_pun_self_mutation_enabled),
    )
    setfield!(old_params, :use_bipenal, getfield(new_params, :use_bipenal))
    setfield!(old_params, :output_save_tick, getfield(new_params, :output_save_tick))

    nothing
end

end # module SimulationParameters
