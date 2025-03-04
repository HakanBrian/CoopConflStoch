using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation:
    SimulationParameter, run_sim_r, run_sim_rep, run_sim_rip, run_sim_rgs


###############################
# Run Simulation
###############################

base_params = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 100000,
    population_size = 500,
    group_size = 10,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)

run_simulation(
    base_params,
    save_file = true,
    filepath = "data/default/default.csv",
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
    ),
)

run_simulation(
    base_params,
    save_file = true,
    filepath = "data/default/default.csv",
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
)

run_simulation(
    base_params,
    save_file = true,
    filepath = "data/default/default.csv",
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :int_pun_ext0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
    ),
    linked_params = Dict(:int_pun_self0 => :int_pun_ext0),
)

run_simulation(
    base_params,
    num_replicates = 20,
    save_file = true,
    filepath = "data/default/default.csv",
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.1)),
        :group_size =>
            [collect(range(5, 50, step = 5))..., collect(range(50, 500, step = 50))...],
    ),
)
