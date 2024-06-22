using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Population Construction
##################

my_parameter = Simulation_Parameters(0.5, 0.5, 0.5, 0.0, 100, 5, 100000, 0.0, 0.1, 0.0, 0.004, 10);
my_population = population_construction(my_parameter);


##################
# Behav Eq
##################

# Define starting parameters
individual1 = Individual(0.2, 0.4, 0.1, 0.5, 0, 0);
individual2 = Individual(0.3, 0.5, 0.2, 0.5, 0, 0);
pair = [(individual1, individual2)];
norm = mean([individual1.a, individual2.a])
punishment = mean([individual1.p, individual2.p])

individual3 = Individual(0.4, 0.2, 0.5, 0.7, 0, 0);
individual4 = Individual(0.5, 0.3, 0.6, 0.8, 0, 0);
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
my_parameter = Simulation_Parameters(0.5, 0.5, 0.5, 0.0, 10, 5, 1000, 0.0, 0.0, 0.0, 0.0, 1); # N has to be a multiple of 4
my_population = population_construction(my_parameter)

individuals1 = Individual(0.5, 0.5, 0.5, 0.0, 1, 0)
individuals2 = Individual(0.5, 0.5, 0.5, 0.0, 2, 0)
individuals3 = Individual(0.5, 0.5, 0.5, 0.0, 3, 0)
individuals4 = Individual(0.5, 0.5, 0.5, 0.0, 4, 0)

# Bootstrap to increase sample size
for i in 1:4:(my_parameter.N-3)
    for j in 1:249
        set_individual!(my_population, i, individuals1)
        set_individual!(my_population, i+1, individuals2)
        set_individual!(my_population, i+2, individuals3)
        set_individual!(my_population, i+3, individuals4)
    end
end

# Ensure 250 copies of each parent
println("Initial population with payoff 4: ", count(payoff -> payoff == 4, my_population.payoffs))

# Complete a round of reproduction
reproduce!(my_population)

# Offspring should have parent 4 as their parent ~40% of the time
println("New population with payoff 4: ", count(payoff -> payoff == 4, my_population.payoffs))


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