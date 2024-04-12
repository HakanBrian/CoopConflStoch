include("CoopConflGameStructs.jl")
include("CoopConflGameFuncs.jl")

##################
# population_construction
##################

my_parameter = simulation_parameters(20,5000,11,0.7,0.05,0.45,0.5,0.4,0.0)

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

# ModelingToolkit
# Need to get the derivative of objective for it to work properly
using ModelingToolkit, OrdinaryDiffEq, Optimization, OptimizationOptimJL
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables action1(t) action2(t)
@parameters a1 a2 p1 p2 T1 T2

eqs = [D(action1) ~ objective(action1, action2, a1, a2, p1, p2, T1)
       D(action2) ~ objective(action2, action1, a2, a1, p2, p1, T2)]

@mtkbuild sys = ODESystem(eqs, t)
prob = ODEProblem(sys, [action1 => 0.2, action2 => 0.3], (0, 20), [a1 => 0.4, a2 => 0.5, p1 => 0.1, p2 => 0.2, T1 => 0.5, T2 => 0.5])
sol = solve(prob, Tsit5())


# Not working
behav(x, p) = [objective(x[1], x[2], p[1], p[2], p[3], p[4], p[5])
                objective(x[2], x[1], p[1], p[2], p[3], p[4], p[6])]

u0 = [0.2, 0.3]
p = [0.4, 0.4, 0.5, 0.5, 0.6, 0.7]

optf = OptimizationFunction(behav, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, p, lb = [0, 0], ub = [1, 1])
sol = solve(prob, NelderMead())

behav_eq!(individual(0.2, 0.4, 0.1, 0.5, 0, 0), individual(0.3, 0.5, 0.2, 0.5, 0, 0))