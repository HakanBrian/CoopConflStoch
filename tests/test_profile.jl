using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation


############
# Profiling #####################################################################################################################
############

params = MainSimulation.SimulationParameter()  # uses all default values
population = MainSimulation.population_construction(params);

# compilation
@time MainSimulation.simulation(population);
# pure runtime
@profview @time MainSimulation.simulation(population);


#############
# Group size ####################################################################################################################
#############

# Test group size 10
parameter_10 = MainSimulation.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 50,
    group_size = 10,
    relatedness = 0.5,
);
population_10 = MainSimulation.population_construction(parameter_10);
@time simulation_10 = MainSimulation.simulation(population_10);
@profview @time simulation_10 = MainSimulation.simulation(population_10);

# Test group size 20
parameter_20 = MainSimulation.SimulationParameter(
    action0 = 0.1f0,
    norm0 = 2.0f0,
    ext_pun0 = 0.1f0,
    int_pun_ext0 = 0.0f0,
    int_pun_self0 = 0.0f0,
    population_size = 50,
    group_size = 20,
    relatedness = 0.5,
);
population_20 = MainSimulation.population_construction(parameter_20);
@time simulation_20 = MainSimulation.simulation(population_20);
@profview @time simulation_20 = MainSimulation.simulation(population_20);
