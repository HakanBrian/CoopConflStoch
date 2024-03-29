include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

##################
# population_construction
##################

my_parameter = simulation_parameters(20,5000,10,0.7,0.45,0.5,0.4,0.0)

my_population = population_construction(my_parameter)

my_population.individuals[867]

social_interactions!(my_population)

my_population.pairings


##################
# pairing & payoff
##################

individuals_dict = Dict{Int64, agent}()
individuals_dict[1] = agent(1, 0.4, 0.67, 0.54, 0, 0, 0, 0)
individuals_dict[2] = agent(2, 0.6, 0.36, 0.45, 0, 0, 0, 0)
individuals_dict[3] = agent(3, 0.5, 0.75, 0.63, 0, 0, 0, 0)
individuals_dict[4] = agent(4, 0.3, 0.24, 0.43, 0, 0, 0, 0)

my_pairing = [pair(individuals_dict[1], individuals_dict[2]), pair(individuals_dict[3], individuals_dict[4])]

payoff!(my_pairing[1])
payoff!(my_pairing[2])

individuals_dict

individuals_copy = collect(values(copy(individuals_dict)))
shuffle!(individuals_copy)

individuals_copy[2]
my_population.pairings = []

for i in 1:2:(length(individuals_copy)-1)
    push!(my_population.pairings, pair(individuals_copy[i], individuals_copy[i+1]))
end

my_population.pairings
my_population.pairings[1].individual1.id
my_population.individuals
my_population.individuals[my_population.pairings[1].individual1.id] = my_population.pairings[1].individual1