include("NetworkGameStructs.jl")
include("NetworkGameFuncs.jl")

my_parameter = simulation_parameters(20,5000,10,0.45,0.5,0.4,0.0)

my_population = population_construction(my_parameter)

my_interactions = social_interactions!(my_population)