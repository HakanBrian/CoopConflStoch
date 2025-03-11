using BenchmarkTools, Revise

push!(LOAD_PATH, abspath("src"));
using MainSimulation

# Apply changes
Revise.revise()


######################
# SimulationParameter ###########################################################################################################
######################

base_params = MainSimulation.SimulationParameter()

sweep_vars = Dict{Symbol,Vector{<:Real}}(
    :relatedness => [0.0, 1.0],
    :int_pun_ext0 => Float32[0.0, 3.0, 6.0],
    :group_size => [5, 50, 500],
)

linked_params = Dict(:int_pun_ext0 => :group_size, :int_pun_self0 => :int_pun_ext0)

MainSimulation.generate_params(base_params, sweep_vars, linked_params)
