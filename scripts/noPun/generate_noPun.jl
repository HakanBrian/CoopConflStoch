using Distributed


###############################
# Load MainSimulation
###############################

# @everywhere cd("")  # Set dir to base
@everywhere push!(LOAD_PATH, abspath("src"))
@everywhere using MainSimulation


###############################
# Run Simulation
###############################

base_params = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 10,
    ext_pun_mutation_enabled = false,
    int_pun_ext_mutation_enabled = false,
    int_pun_self_mutation_enabled = false,
    output_save_tick = 10,
)

# r
run_simulation(
    base_params,
    save_file = true,
    filepath = "data/noPun/r",
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
    ),
)

# rep/rip
run_simulation(
    rep_params,
    save_file = true,
    filepath = "data/noPun/reip",
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => zeros(Float32, 21),
    ),
)

# rgs
run_simulation(
    base_params,
    num_replicates = 20,
    save_file = true,
    filepath = "data/noPun/rgs",
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.1)),
        :group_size =>
            [collect(range(5, 50, step = 5))..., collect(range(100, 500, step = 50))...],
    ),
)

# rDiffGS
run_simulation(
    base_params_rdgs,
    filepath = "data/noPun/rDiffGS",
    save_file = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
        :group_size => [5, 50, 500],
    ),
)

# repDiffGS
run_simulation(
    base_params_rdgs,
    num_replicates = 20,
    save_file = true,
    filepath = "data/noPun/repDiffGS",
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => zeros(Float32, 21),
        :group_size => [5, 50, 500],
    ),
)

#ripDiffGS
run_simulation(
    base_params_rdgs,
    num_replicates = 20,
    save_file = true,
    filepath = "data/noPun/ripDiffGS",
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :int_pun_ext0 => zeros(Float32, 101),
        :group_size => [10, 50, 500],
    ),
    linked_params = Dict(:int_pun_self0 => :int_pun_ext0),
)
