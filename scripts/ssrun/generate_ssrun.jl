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

base_param = SimulationParameter(
    action0 = 0.0f0,
    norm0 = 0.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 2000000,
    population_size = 500,
    group_size = 10,
    relatedness = 0.5,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 20,
)

run_simulation(
    base_param,
    filepath = "data/ssrun/ext_int_pun/ext_int_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :norm0 => Float32[0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 7.0],
    ),
)

nep_param = update_params(base_param, ext_pun_mutation_enabled = false)
run_simulation(
    nep_param,
    filepath = "data/ssrun/no_ext_pun/no_ext_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :norm0 => Float32[0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 7.0],
    ),
)

nip_param = update_params(base_param, int_pun_ext_mutation_enabled = false, int_pun_self_mutation_enabled = false)
run_simulation(
    nip_param,
    filepath = "data/ssrun/no_int_pun/no_int_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :norm0 => Float32[0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 7.0],
    ),
)
