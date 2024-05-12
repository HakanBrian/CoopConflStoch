include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

##################
# population_construction
##################

my_parameter = simulation_parameters(20,11,0.7,0.05,0.45,0.5,0.4,0.0)

action0_dist = Truncated(Normal(0.45, 0.05), 0, 1)

rand(action0_dist)

my_population = population_construction(my_parameter)

my_population.individuals[8]

social_interactions!(my_population)

my_population.individuals


##################
# pairing & payoff
##################

individuals_dict = Dict{Int64, individual}()
individuals_dict[1] = individual(0.4, 0.67, 0.54, 0, 0, 0)
individuals_dict[2] = individual(0.6, 0.36, 0.45, 0, 0, 0)
individuals_dict[3] = individual(0.5, 0.75, 0.63, 0, 0, 0)
individuals_dict[4] = individual(0.3, 0.24, 0.43, 0, 0, 0)

individuals_dict

individuals_key = collect(keys(copy(individuals_dict)))
individuals_shuffle = shuffle(individuals_key)

individuals_shuffle[2]

if length(individuals_dict) % 2 != 0
    push!(individuals_shuffle, individuals_key[rand(1:length(individuals_dict))])
end

for i in 1:2:(length(individuals_shuffle)-1)
    total_payoff!(individuals_dict[individuals_shuffle[i]], individuals_dict[individuals_shuffle[i+1]])
end

individuals_dict

payoffs = [individual.payoff for individual in values(individuals_dict)]


##################
# Optimization
##################

@variables action1(t) action2(t)
@parameters a1 a2 p1 p2 T1 T2

function objective_derivative(action, other_action, a, other_a, p, other_p, T)
    return ForwardDiff.derivative(action -> objective(action, other_action, a, other_a, p, other_p, T), action)
end

eqs = [D(action1) ~ objective_derivative(action1, action2, a1, a2, p1, p2, T1)
       D(action2) ~ objective_derivative(action2, action1, a2, a1, p2, p1, T2)]

@mtkbuild sys = ODESystem(eqs, t)
prob = ODEProblem(sys, [action1 => 0.2, action2 => 0.3], (0, 20), [a1 => 0.4, a2 => 0.5, p1 => 0.1, p2 => 0.2, T1 => 0.5, T2 => 0.5])
sol = solve(prob, Tsit5())


##################
# behav_eq
##################

# define starting parameters
individual1 = individual(0.2, 0.4, 0.1, 0.5, 0, 0)
individual2 = individual(0.3, 0.5, 0.2, 0.5, 0, 0)

# calculate behave eq
behav_eq!(individual1, individual2)

# Compare values with mathematica code
individual1  # should be around 0.41303
individual2  # individual 1 and 2 should have nearly identical values


##################
# reproduce
##################

# Have the population interact
social_interactions!(my_population)

# Calculate the payoffs after interaction
my_population
payoffs = [individual.payoff for individual in values(my_population.individuals)]

# Reproduce after payoffs calculated
my_population.mean_w = mean(payoffs)
genotype_array = sample(1:my_population.parameters.N, ProbabilityWeights(payoffs), my_population.parameters.N, replace=true)
old_individuals = copy(my_population.individuals)
for (res_i, offspring_i) in zip(1:my_population.parameters.N, genotype_array)
    my_population.individuals[res_i] = old_individuals[genotype_array[offspring_i]]
end

# Check to see if the offspring are correctly selected
old_individuals
my_population.individuals

# same as before but through the completed function
reproduce!(my_population)  # compare with previous results to ensure the function in use is correct


##################
# individual copy
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
# mutation
##################

mutate!(my_population)

my_population.individuals