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

copy_individuals_dict = copy(original_individuals_dict)

# Modify some fields in the original and copied objects
original_individuals_dict[1].action = 0.9
original_individuals_dict[2].action = 0.8
copy_individuals_dict[1].action = 0.1
copy_individuals_dict[2].action = 0.2

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


##################
# Population Copy
##################

old_new_individuals_dict = Dict{Int64, individual}()
old_new_individuals_dict[1] = individual(0.4, 0.67, 0.54, 0, 0, 0)
old_new_individuals_dict[2] = individual(0.6, 0.36, 0.45, 0, 0, 0)

old_old_individuals_dict = Dict{Int64, individual}()
old_old_individuals_dict[1] = individual(0.5, 0.27, 0.64, 0.1, 0.3, 0)
old_old_individuals_dict[2] = individual(0.65, 0.26, 0.75, 0.53, 0.12, 0)

old_population = population(simulation_parameters(0.45,0.5,0.4,0.0,20,15,11,0.0,0.7,0.05,0.05,20), old_new_individuals_dict, old_old_individuals_dict)

new_new_individuals_dict = Dict{Int64, individual}()
new_new_individuals_dict[1] = individual(0.3, 0.57, 0.24, 0.6, 0.4, 0)
new_new_individuals_dict[2] = individual(0.6, 0.36, 0.45, 0.7, 0.44, 0)

new_old_individuals_dict = Dict{Int64, individual}()
new_old_individuals_dict[1] = individual(0.54, 0.27, 0.66, 0.12, 0.56, 0)
new_old_individuals_dict[2] = individual(0.25, 0.98, 0.36, 0.86, 0.86, 0)

new_population = population(simulation_parameters(0.15,0.15,0.34,0.2,21,15,12,0.0,0.3,0.45,0.2,20), new_new_individuals_dict, new_old_individuals_dict)

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

my_parameter = simulation_parameters(0.6, 0.5, 0.45, 0.0, 50, 20, 10, 0.0, 0.05, 0.05, 0.05, 1)
my_population = population_construction(my_parameter)


##################
# Behav Eq
##################

# Define starting parameters
individual1 = individual(0.2, 0.4, 0.1, 0.5, 0, 0)
individual2 = individual(0.3, 0.5, 0.2, 0.5, 0, 0)

# Calculate behave eq
behav_eq!(individual1, individual2, my_parameter.tmax, my_parameter.v)

# Compare values with mathematica code
individual1  # should be around 0.41303
individual2  # individual 1 and 2 should have nearly identical values


##################
# Social Interactions
##################

social_interactions!(my_population)


##################
# Reproduce
##################

# Create sample population
my_parameter = simulation_parameters(0.5, 0.5, 0.5, 0.0, 10, 20, 4, 0.0, 0.0, 0.0, 0.0, 1)
individuals_dict = Dict{Int64, individual}()
old_individuals_dict = Dict{Int64, individual}()
my_population = population(my_parameter, individuals_dict, old_individuals_dict)

my_population.individuals[1] = individual(0.5, 0.5, 0.5, 0.0, 1, 0)
my_population.individuals[2] = individual(0.5, 0.5, 0.5, 0.0, 2, 0)
my_population.individuals[3] = individual(0.5, 0.5, 0.5, 0.0, 3, 0)
my_population.individuals[4] = individual(0.5, 0.5, 0.5, 0.0, 4, 0)

# Bootstrap to increase sample size
original_size = length(my_population.individuals)
new_key = original_size + 1
for i in 1:original_size
    for j in 1:24
        my_population.individuals[new_key] = copy(my_population.individuals[i])
        new_key += 1
    end
end

# Ensure 25 of each copies of each parent
println("Initial population with payoff 4: ", count(individual -> individual.payoff == 4, values(my_population.individuals)))

# Complete a round of reproduction
my_population.old_individuals = copy(my_population.individuals)
reproduce!(my_population)

# Offspring should have parent 1 as their parent ~40% of the time
println("New population with payoff 4: ", count(individual -> individual.payoff == 4, values(my_population.individuals)))


##################
# Mutate
##################

mutate!(my_population)


##################
# Profiling
##################

# compilation
@profview simulation(my_population);
# pure runtime
@profview simulation(my_population);

@profview @btime social_interactions!(my_population)

@profview @btime reproduce!(my_population);

@profview @btime mutate!(my_population);
