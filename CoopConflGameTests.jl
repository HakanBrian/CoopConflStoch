using BenchmarkTools


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Parameter Copy
##################

old_parameters = simulation_parameters(0.45,0.5,0.4,0.0,20,15,11,0.0,0.7,0.05,0.05,20)
new_parameters = simulation_parameters(0.15,0.15,0.34,0.2,21,15,12,0.0,0.3,0.45,0.1,20)

# Perform copying operations
copied_parameters = copy(new_parameters)
copy!(old_parameters, new_parameters)

# Modify some fields in the original and copied objects
new_parameters.N = 423
old_parameters.N = 4
copied_parameters.N = 2

# Check if modifications affect the copied individual
println(old_parameters.N == 423)  # Should print false
println(old_parameters.N == 4)  # Should print true

println(copied_parameters.N == 423)  # Should print false
println(copied_parameters.N == 2)  # Should print true


##################
# Individual Copy
##################

original_individual = individual(0.5, 0.4, 0.3, 0.2, 0.0, 0.0)
copied_individual = copy(original_individual)

# Modify some fields in the original and copied objects
original_individual.action = 0.9
copied_individual.action = 0.1

# Check if modifications affect the original individual
println(original_individual.action == 0.9)  # Should print true
println(original_individual.action == 0.1)  # Should print false

# Check if modifications affect the copied individual
println(copied_individual.action == 0.9)  # Should print false
println(copied_individual.action == 0.1)  # Should print true


original_individuals_dict = Dict{Int64, individual}()
original_individuals_dict[1] = individual(0.4, 0.67, 0.54, 0, 0, 0)
original_individuals_dict[2] = individual(0.6, 0.36, 0.45, 0, 0, 0)

old_individuals_dict = Dict{Int64, individual}()
old_individuals_dict[1] = individual(0.43, 0.5, 0.34, 0, 0, 0)
old_individuals_dict[2] = individual(0.36, 0.3, 0.55, 0, 0, 0)

copy_individuals_dict = copy(original_individuals_dict)
copy!(old_individuals_dict, original_individuals_dict)

# Modify some fields in the original and copied objects
original_individuals_dict[1].action = 0.9
original_individuals_dict[2].action = 0.8
copy_individuals_dict[1].action = 0.1
copy_individuals_dict[2].action = 0.2
old_individuals_dict[1].action = 0.4
old_individuals_dict[2].action = 0.5

# Check if modifications affect the original individual
println(original_individuals_dict[1].action == 0.9)  # Should print true
println(original_individuals_dict[1].action == 0.1)  # Should print false
println(original_individuals_dict[2].action == 0.8)  # Should print true
println(original_individuals_dict[2].action == 0.2)  # Should print false

# Check if modifications affect the copied individual
println(copy_individuals_dict[1].action == 0.1)  # Should print true
println(copy_individuals_dict[1].action == 0.9)  # Should print false
println(copy_individuals_dict[2].action == 0.2)  # Should print true
println(copy_individuals_dict[2].action == 0.8)  # Should print false

# Check if modifications affect the old individual
println(old_individuals_dict[1].action == 0.4)  # Should print true
println(old_individuals_dict[1].action == 0.9)  # Should print false
println(old_individuals_dict[2].action == 0.5)  # Should print true
println(old_individuals_dict[2].action == 0.8)  # Should print false


##################
# Population Copy
##################

old_new_individuals_dict = Dict{Int64, individual}()
old_new_individuals_dict[1] = individual(0.4, 0.67, 0.54, 0, 0, 0)
old_new_individuals_dict[2] = individual(0.6, 0.36, 0.45, 0, 0, 0)

old_population = population(simulation_parameters(0.45,0.5,0.4,0.0,20,15,11,0.0,0.7,0.05,0.05,20), old_new_individuals_dict, 0, 0)

new_new_individuals_dict = Dict{Int64, individual}()
new_new_individuals_dict[1] = individual(0.3, 0.57, 0.24, 0.6, 0.4, 0)
new_new_individuals_dict[2] = individual(0.6, 0.36, 0.45, 0.7, 0.44, 0)

new_population = population(simulation_parameters(0.15,0.15,0.34,0.2,21,15,12,0.0,0.3,0.45,0.2,20), new_new_individuals_dict, 0, 0)

# Perform copying operations
copied_population = copy(new_population)
copy!(old_population, new_population)

# Modify some fields in the original and copied objects
new_population.individuals[1].action = 0.0
old_population.individuals[1].action = 0.1
copied_population.individuals[1].action = 0.2

new_population.parameters.N = 342
old_population.parameters.N = 3
copied_population.parameters.N = 4

# Check if modifications affect the copied individual
println(old_population.individuals[1].action == 0.0)  # Should print false
println(old_population.individuals[1].action == 0.1)  # Should print true

println(old_population.parameters.N == 342)  # Should print false
println(old_population.parameters.N == 3)  # Should print true

println(copied_population.individuals[1].action == 0.0)  # Should print false
println(copied_population.individuals[1].action == 0.2)  # Should print true

println(copied_population.parameters.N == 342)  # Should print false
println(copied_population.parameters.N == 4)  # Should print true


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
behav_eq_MTK!(pair, norm, punishment, my_parameter.tmax, my_parameter.v)

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

mutate!(my_population)


##################
# Profiling
##################

# compilation
@btime simulation(my_population);
# pure runtime
@profview simulation(my_population);