using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Population Construction
##################

params = SimulationParameters()  # uses all default values
my_population = population_construction(params);

##################
# Behav Eq
##################

# Define starting parameters
individual1 = Individual(0.2, 0.4, 0.1, 0.5, 0.0, 0);
individual2 = Individual(0.3, 0.5, 0.2, 0.5, 0.0, 0);
pair = [(individual1, individual2)];
norm = mean([individual1.a, individual2.a])
punishment = mean([individual1.p, individual2.p])

# Calculate behave eq
behav_eq!(pair, norm, punishment, 10.0, 0.0)

# Compare values with mathematica code
individual1  # should be around 0.413
individual2  # individual 1 and 2 should have nearly identical values


##################
# Fitness and Payoff
##################

individual1 = Individual(0.1, 2.0, 0.1, 0.0, 0.0, 0);
individual2 = copy(individual1);
pair = [(individual1, individual2)];
norm = mean([individual1.a, individual2.a])
punishment = mean([individual1.p, individual2.p])

behav_eq!(pair, norm, punishment, 30.0, 0.0)

total_payoff!(individual1, 0.0)

fitness(individual1, 50)


##################
# Social Interactions
##################

social_interactions!(my_population)

println(my_population.individuals)


##################
# Reproduce
##################

# To use this test payoffs need to be copied into the next generation

# Create sample population
my_parameter = SimulationParameters(p0=0.0, gmax=10, population_size=1000, mutation_rate=0.0)
individuals_dict = Dict{Int64, Individual}()
my_population = Population(my_parameter, individuals_dict, 0, 0)

my_population.individuals[1] = Individual(0.5, 0.5, 0.0, 0.0, 1, 0)
my_population.individuals[2] = Individual(0.5, 0.5, 0.0, 0.0, 2, 0)
my_population.individuals[3] = Individual(0.5, 0.5, 0.0, 0.0, 3, 0)
my_population.individuals[4] = Individual(0.5, 0.5, 0.0, 0.0, 4, 0)

# Bootstrap to increase sample size
original_size = length(my_population.individuals)
new_key = original_size + 1
for i in 1:original_size
    for j in 1:249
        my_population.individuals[new_key] = copy(my_population.individuals[i])
        new_key += 1
    end
end

# Ensure 250 copies of each parent
println("Initial population with payoff 4: ", count(individual -> individual.payoff == 4, values(my_population.individuals)))

# Complete a round of reproduction
reproduce!(my_population)

# Offspring should have parent 4 as their parent ~40% of the time (only if there is no scaling)
println("New population with payoff 4: ", count(individual -> individual.payoff == 4, values(my_population.individuals)))


##################
# Mutate
##################

mutate!(my_population, truncation_bounds(my_population.parameters.mutation_variance, 0.99))

println(my_population)


##################
# Profiling
##################

# compilation
@time simulation(my_population);
# pure runtime
@time simulation(my_population);