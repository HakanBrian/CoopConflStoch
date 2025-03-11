using Distributed


###############################
# Load MainSimulation
###############################

# @everywhere cd("")  # Set dir to base

@everywhere begin
    push!(LOAD_PATH, abspath("src"))
    using MainSimulation
end


###############################
# Run Simulation
###############################

base_params_rdgs = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 5,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)

run_simulation(
    base_params_rdgs,
    filepath = "data/rDiffGS/rDiffGS",
    save_file = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
        :group_size => [5, 50, 500],
    ),
)


unipenal_params_rdgs = update_params(base_params_rdgs, use_bipenal = false)

run_simulation(
    unipenal_params_rdgs,
    filepath = "data/rDiffGS/rDiffGS",
    save_file = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
        :group_size => [5, 50, 500],
    ),
)
