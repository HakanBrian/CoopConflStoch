include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

my_parameter = simulation_parameters(20,5000,1000,0.7,0.45,0.5,0.4,0.0)

my_population = population_construction(my_parameter)

my_interactions = social_interactions!(my_population)

my_population.individuals[867]