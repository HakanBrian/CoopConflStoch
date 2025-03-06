using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, run_simulation


###############################
# Run Simulation
###############################

base_params_rdgs = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 5,
    ext_pun_mutation_enabled = false,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = false,
    output_save_tick = 10,
)

run_simulation(
    base_params,
    filepath = "data/ripDiffGS/ripDiffGS",
    save_file = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :int_pun_ext0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
        :group_size => [5, 50, 500],
    ),
    linked_params = Dict(:int_pun_self0 => :int_pun_ext0),
)
