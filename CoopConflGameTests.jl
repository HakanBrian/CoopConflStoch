using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Population Construction
##################

my_parameter = simulation_parameters(0.5, 0.5, 0.5, 0.0, 100, 5, 100000, 0.0, 0.1, 0.0, 0.004, 10);
my_population = population_construction(my_parameter);


##################
# Behav Eq
##################

# Define starting parameters
individual1 = individual(0.2, 0.4, 0.1, 0.5, 0, 0);
individual2 = individual(0.3, 0.5, 0.2, 0.5, 0, 0);
pair = [(individual1, individual2)];
norm = mean([individual1.a, individual2.a])
punishment = mean([individual1.p, individual2.p])

individual3 = individual(0.4, 0.2, 0.5, 0.7, 0, 0);
individual4 = individual(0.5, 0.3, 0.6, 0.8, 0, 0);
pair_all = [(individual1, individual2), (individual3, individual4)];
norm_all = mean([individual1.a, individual2.a, individual3.a, individual4.a])
punishment_all = mean([individual1.p, individual2.p, individual3.p, individual4.p])

# Calculate behave eq
@time behav_eq!(pair, norm, punishment, my_parameter.tmax, my_parameter.v)
@time behav_eq!(pair_all, norm_all, punishment_all, my_parameter.tmax, my_parameter.v)

# Compare values with mathematica code
individual1  # should be around 0.41303
individual2  # individual 1 and 2 should have nearly identical values
individual3  # should be around 0.32913
individual4  # individual 3 and 4 should have nearly identical values


##################
# Social Interactions
##################

@btime social_interactions!(my_population)


##################
# Reproduce
##################

# Create sample population
my_parameter = simulation_parameters(0.5, 0.5, 0.5, 0.0, 10, 5, 1000, 0.0, 0.0, 0.0, 0.0, 1)
individuals_dict = Dict{Int64, individual}()
my_population = population(my_parameter, individuals_dict, 0, 0)

my_population.individuals[1] = individual(0.5, 0.5, 0.5, 0.0, 1, 0)
my_population.individuals[2] = individual(0.5, 0.5, 0.5, 0.0, 2, 0)
my_population.individuals[3] = individual(0.5, 0.5, 0.5, 0.0, 3, 0)
my_population.individuals[4] = individual(0.5, 0.5, 0.5, 0.0, 4, 0)

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

# Offspring should have parent 4 as their parent ~40% of the time
println("New population with payoff 4: ", count(individual -> individual.payoff == 4, values(my_population.individuals)))


##################
# Mutate
##################

mutate!(my_population, truncation_bounds(my_population.parameters.mut_var, 0.99))

println(my_population)


##################
# Profiling
##################

# compilation
@btime simulation(my_population);
# pure runtime
@profview @time simulation(my_population);