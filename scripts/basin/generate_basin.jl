using Distributed


###############################
# Load MainSimulation
###############################

# @everywhere cd("")  # Set dir to base

@everywhere include(joinpath(pwd(), "src", "Main.jl"))
@everywhere using .MainSimulation
@everywhere import .MainSimulation: SimulationParameter, update_params, run_simulation


###############################
# Run Simulation
###############################

sweep_rgs = Dict{Symbol,Vector{<:Real}}(
    :relatedness => collect(range(0, 1.0, step = 0.25)),
    :group_size => [5, 50, 500],
);

base_param = SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.0f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    generations = 200000,
    population_size = 500,
    group_size = 5,
    relatedness = 0.0,
    ext_pun_mutation_enabled = true,
    int_pun_ext_mutation_enabled = true,
    int_pun_self_mutation_enabled = true,
    output_save_tick = 20,
)

# external and bipenal internal punishment
run_simulation(
    base_param,
    filepath = "data/basin/ext_int_pun/ext_int_pun",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# external and unipenal internal punishment
unipenal_param = update_params(base_param, use_bipenal = false)
run_simulation(
    unipenal_param,
    filepath = "data/basin/ext_int_pun/ext_int_pun",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# fixed external punishment
fixed_ext_pun = update_params(base_param, ext_pun_mutation_enabled = false)
run_simulation(
    fixed_ext_pun,
    filepath = "data/basin/fixed_ext_pun/fixed_ext_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.25)),
        :ext_pun0 => Float32[0.1, 0.5, 1.5],
        :group_size => [5, 50, 500],
    ),
    linked_params = Dict(:ext_pun0 => :group_size),
)

# fixed internal punishment
fixed_int_pun = update_params(
    base_param,
    int_pun_ext_mutation_enabled = false,
    int_pun_self_mutation_enabled = false,
)
run_simulation(
    fixed_int_pun,
    filepath = "data/basin/fixed_int_pun/fixed_int_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.25)),
        :int_pun_ext0 => Float32[0.1, 0.5, 1.5],
        :group_size => [5, 50, 500],
    ),
    linked_params = Dict(:int_pun_ext0 => :group_size, :int_pun_self0 => :int_pun_ext0),
)

# fixed norm
fixed_norm = update_params(base_param, norm_mutation_enabled = false)
run_simulation(
    fixed_norm,
    filepath = "data/basin/fixed_norm/fixed_norm",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# higher external punishment
run_simulation(
    base_param,
    filepath = "data/basin/higher_ext_pun/higher_ext_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.25)),
        :ext_pun0 => Float32[0.1, 0.5, 1.5],
        :group_size => [5, 50, 500],
    ),
    linked_params = Dict(:ext_pun0 => :group_size),
)

# higher internal punishment
run_simulation(
    base_param,
    filepath = "data/basin/higher_int_pun/higher_int_pun",
    sweep_full = true,
    sweep_vars = Dict{Symbol,Vector{<:Real}}(
        :relatedness => collect(range(0, 1.0, step = 0.25)),
        :int_pun_ext0 => Float32[0.1, 0.5, 1.5],
        :group_size => [5, 50, 500],
    ),
    linked_params = Dict(:int_pun_ext0 => :group_size, :int_pun_self0 => :int_pun_ext0),
)

# no external punishment
no_ext_pun = update_params(base_param, ext_pun_mutation_enabled = false)
run_simulation(
    no_ext_pun,
    filepath = "data/basin/no_ext_pun/no_ext_pun",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# no internal punishment
no_int_pun = update_params(
    base_param,
    int_pun_ext_mutation_enabled = false,
    int_pun_self_mutation_enabled = false,
)
run_simulation(
    no_int_pun,
    filepath = "data/basin/no_int_pun/no_int_pun",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# no punishment
no_pun = update_params(
    base_param,
    ext_pun_mutation_enabled = false,
    int_pun_ext_mutation_enabled = false,
    int_pun_self_mutation_enabled = false,
)
run_simulation(
    no_pun,
    filepath = "data/basin/no_pun/no_pun",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)

# stochastic initial values
stoch_init = update_params(base_param, trait_variance = 0.1)
run_simulation(
    stoch_init,
    filepath = "data/basin/stoch_init/stoch_init",
    sweep_full = true,
    sweep_vars = sweep_rgs,
)
