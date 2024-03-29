include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

##################
# population_construction
##################

my_parameter = simulation_parameters(20,5000,2,0.7,0.45,0.5,0.4,0.0)

my_population = population_construction(my_parameter)

my_population.individuals[867]


##################
# pairing & payoff
##################

individuals_dict = Dict{Int64, agent}()
individuals_dict[1] = agent(1, 0.4, 0, 0, 0, 0, 0, 0)
individuals_dict[2] = agent(2, 0.6, 0, 0, 0, 0, 0, 0)

my_pairing = pair(individuals_dict[1],individuals_dict[2])

payoff(my_pairing)

individuals_dict