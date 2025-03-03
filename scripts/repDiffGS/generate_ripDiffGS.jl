using Distributed


###############################
# Load MainSimulation
###############################

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, run_sim_rep


###############################
# Run Simulation
###############################

base_params_rdgs_base = SimulationParameter(
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

run_sim_all(
    base_params,
    filepath = "data/repDiffGS/rep2",
    save_file = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :relatedness => collect(range(0, 1.0, step = 0.05)),
        :ext_pun0 => collect(range(0.0f0, 1.0f0, step = 0.05f0)),
        :group_size => [5, 50, 500]
    ),
    save_generations = [0.25, 0.5, 0.75, 1.0],
)
