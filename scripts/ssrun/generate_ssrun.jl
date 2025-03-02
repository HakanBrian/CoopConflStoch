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

base_param = SimulationParameters(
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
    int_pun_ext_mutation_enabled = false,
    int_pun_self_mutation_enabled = false,
    output_save_tick = 20,
)

run_sim_all(
    base_param,
    filename = "data/ssrun/ssbase",
    sweep_full = true,
    sweep_vars = Dict{Symbol,AbstractVector}(
        :norm0 => Float32[0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 7.0],
    ),
)
