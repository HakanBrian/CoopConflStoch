using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


######################
# SimulationParameter ###########################################################################################################
######################

base_params = MainSimulation.SimulationParameter()

sweep_vars = Dict(
    :relatedness => [0.0, 1.0],
    :int_pun_ext0 => Float32[0.0, 3.0],
    :group_size => [5, 50, 500]
)

parameters = vec([
    MainSimulation.update_params(base_params; NamedTuple{Tuple(keys(sweep_vars))}(values)...)
    for values in Iterators.product(values(sweep_vars)...)
])

linked_params = Dict(:int_pun_self0 => :int_pun_ext0)

valid_linked_parameters = Dict(
    k => v for (k, v) in linked_params if v in keys(sweep_vars)
)

# Store only primary parameters for iteration
primary_keys = collect(setdiff(keys(sweep_vars), keys(valid_linked_parameters)))

MainSimulation.generate_params(base_params, sweep_vars, linked_params)

# param gen logic
parameters = vec([
    MainSimulation.update_params(base_params;
        NamedTuple{Tuple(primary_keys)}(values)...,
        Dict(
            k => values[findfirst(==(v), primary_keys)] for (k, v) in valid_linked_parameters
            if findfirst(==(v), primary_keys) !== nothing
        )...
    )
    for values in Iterators.product((sweep_vars[k] for k in primary_keys)...)
])

#suffix generation
for values in Iterators.product(values(sweep_vars)...)
    param_dict = Dict(k => v for (k, v) in zip(keys(sweep_vars), values))
    println(param_dict)
    suffix = MainSimulation.generate_filename_suffix(param_dict, "Filtered", time_point=10)
    println(suffix)
end

#Task creation
sweep_vars = Dict{Symbol,AbstractVector}(
    :relatedness => collect(range(0, 1.0, step = 0.1)),
    :group_size =>
        [collect(range(5, 50, step = 5))..., collect(range(50, 500, step = 50))...],
)

parameters_sweep = vec([
    MainSimulation.update_params(base_params; NamedTuple{Tuple(keys(sweep_vars))}(values)...)
    for values in Iterators.product(values(sweep_vars)...)
])

tasks = [
    (idx, parameters, replicate) for (idx, parameters) in enumerate(parameters_sweep) for
    replicate in 1:40
]


sweep_vars = Dict{Symbol,AbstractVector}(:relatedness => collect(range(0, 1.0, step = 0.25)));

param_combinations = vec([
    Dict(k => v for (k, v) in zip(keys(sweep_vars), values)) 
    for values in Iterators.product(values(sweep_vars)...)
])
param_id_to_params = Dict(i => param_combinations[i] for i in 1:5)


for i in 1:5
    # Ensure this param_id exists in our mapping
    if i ∉ keys(param_id_to_params)
        @warn "No parameter combination found for param_id $i, skipping."
        continue
    end
    param_dict = param_id_to_params[i]  # Correct mapping

    # Generate the suffix in the same format as `generate_filename_suffix`
    key = MainSimulation.generate_filename_suffix(param_dict, "Full")
    println(key)
end

for values in Iterators.product(values(sweep_vars)...)
    param_dict = Dict(k => v for (k, v) in zip(keys(sweep_vars), values))
end


# Generate all parameter combinations in a vector of dictionaries
parameters = vec([
    Dict(k => v for (k, v) in zip(keys(sweep_vars), values))
    for values in Iterators.product(values(sweep_vars)...)
])

# Extract column names (keys of the dictionaries)
param_keys = collect(keys(sweep_vars))

# Create a DataFrame where each column corresponds to a parameter
param_df = DataFrame()

# Assign each key as a column, extracting its values from all dictionaries
for key in param_keys
    param_df[!, key] = getindex.(parameters, key)
end

println(param_df)


sweep_vars = Dict{Symbol,AbstractVector}(
    :relatedness => collect(range(0, 1.0, step = 0.05)),
    :ext_pun0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
)

# Sort the keys alphabetically
sorted_keys = sort(collect(keys(sweep_vars)))

# Generate all parameter combinations in a vector of dictionaries
param_combinations = vec([
    Dict(k => v for (k, v) in zip(sorted_keys, values))
    for values in Iterators.product((sweep_vars[k] for k in sorted_keys)...)
])


# Create a mapping from `param_id` to its parameter combination
param_id_to_params = Dict(i => param_combinations[i] for i in 1:441)
param_id_to_params[22]

MainSimulation.generate_params(base_params, sweep_vars)
