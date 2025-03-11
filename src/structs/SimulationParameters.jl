module SimulationParameters

export SimulationParameter, diff_from_default, update_params, generate_params

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
    tolerance::Float64  # Behavioral equilibrium params
    population_size::Int64
    group_size::Int64
    synergy::Float32
    relatedness::Float64
    fitness_scaling_factor::Float64
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
    fitness_scaling_factor::Float64 = 10.0,
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
        fitness_scaling_factor,
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

function diff_from_default(
    instance::SimulationParameter;
    ignore_fields::Set{Symbol} = Set([
        :norm_mutation_enabled,
        :ext_pun_mutation_enabled,
        :int_pun_ext_mutation_enabled,
        :use_bipenal,
        :output_save_tick,
        :generations,
        :population_size,
    ]),
)
    default_instance = SimulationParameter()
    return Dict(
        f => getfield(instance, f) for f in fieldnames(SimulationParameter) if
        f ∉ ignore_fields && getfield(instance, f) != getfield(default_instance, f)
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

function resolve_dependency(
    var::Symbol,
    sweep_vars::Dict{Symbol,Vector{<:Real}},
    linked_params::Dict{Symbol,Symbol},
)
    seen = Set{Symbol}()  # Track visited nodes to avoid cycles

    while haskey(linked_params, var)
        if var in seen
            error("Cycle detected in linked parameters involving $var")
        end
        push!(seen, var)

        var = linked_params[var]  # Follow the dependency chain
        if haskey(sweep_vars, var)
            return var  # Found a valid variable with values
        end
    end

    return var  # Return the final resolved variable
end

function generate_params(
    base_params::SimulationParameter,
    sweep_vars::Dict{Symbol,Vector{<:Real}},
    linked_params = Dict{Symbol,Symbol}();
    combo::Bool = false,
)
    # Convert linked_params into a lookup dictionary (independent => dependents)
    linked_groups = Dict{Symbol,Vector{Symbol}}()
    for (dependent, independent) in linked_params
        root_independent = resolve_dependency(independent, sweep_vars, linked_params)
        push!(get!(linked_groups, root_independent, Vector()), dependent)
    end

    # Sort keys alphabetically, excluding linked dependent parameters
    primary_keys = sort(collect(setdiff(keys(sweep_vars), keys(linked_params))))

    # Generate sweep iterables, ensuring dependencies are zipped
    sweep_iterables = []
    for k in primary_keys
        if haskey(linked_groups, k)
            dep_set = sort(collect(linked_groups[k]))  # Ensure ordered dependencies
            push!(
                sweep_iterables,
                collect(
                    zip(
                        sweep_vars[k],
                        collect(
                            zip(
                                (
                                    sweep_vars[d ∈ keys(sweep_vars) ? d :
                                               resolve_dependency(
                                        d,
                                        sweep_vars,
                                        linked_params,
                                    )] for d in dep_set
                                )...,
                            ),
                        ),
                    ),
                ),
            )
        else
            push!(sweep_iterables, sweep_vars[k])
        end
    end

    # Generate parameter combinations while respecting dependencies
    param_combinations = vec([
        merge(
            Dict(
                indep => values[findfirst(==(indep), primary_keys)][1] for
                indep in primary_keys
            ),
            Dict(
                dep => begin
                    sorted_deps = sort(collect(linked_groups[indep]))
                    values[findfirst(==(indep), primary_keys)][2][findfirst(
                        ==(dep),
                        sorted_deps,
                    )]
                end for indep in sort(collect(keys(linked_groups))) for
                dep in sort(collect(linked_groups[indep]))
            ),
        ) for values in Iterators.product(sweep_iterables...)
    ])

    if combo
        return param_combinations
    end

    parameters = vec([
        update_params(base_params; param_combination...) for
        param_combination in param_combinations
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
        fitness_scaling_factor = getfield(parameters, :fitness_scaling_factor),
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
        :fitness_scaling_factor,
        getfield(new_params, :fitness_scaling_factor),
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
