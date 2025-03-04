using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, run_sim_all


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
    group_size = 5,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 10,
)

run_simulation(
    base_params,
    filepath = "data/rDiffGS/rDiffGS",
    save_file = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.01)),
        :group_size => [5, 50, 500],
    ),
)
