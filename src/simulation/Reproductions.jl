module Reproductions

export reproduce!, fitness, fitness_exp, fitness_exp_norm

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population, offspring!

using ..MainSimulation.Exponentials
import ..MainSimulation.Exponentials: Exponential, normalize_exponentials

using StatsBase

@inline function fitness(pop::Population, idx::Int64)
    return pop.payoff[idx] - pop.ext_pun[idx]
end

function fitness_exp(pop::Population, idx::Int64)
    base_fitness = fitness(pop, idx)
    return exp(base_fitness * 10.0)
end

function fitness_exp_norm(pop::Population, idx::Int64)
    base_fitness = fitness(pop, idx)
    return Exponential(base_fitness * 10.0)
end

function reproduce!(pop::Population)
    # Create a list of indices corresponding to individuals
    indices_list = 1:pop.parameters.population_size

    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness_exp_norm(pop, i), indices_list)

    # Sample indices with the given fitness weights
    normalized_probs = normalize_exponentials(fitnesses)
    sampled_indices = sample(
        indices_list,
        Weights(normalized_probs),
        pop.parameters.population_size,
        replace = true,
        ordered = false,
    )

    # Sort sampled indices to avoid unnecessary memory shuffling during offspring generation
    sort!(sampled_indices)

    # Create new offspring from sampled individuals
    for i in 1:pop.parameters.population_size
        offspring!(pop, i, sampled_indices[i])
    end

    nothing
end

#= Nonsybolic version
function reproduce!(pop::Population)
    # Create a list of indices corresponding to individuals
    indices_list = 1:pop.parameters.population_size

    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness_exp(pop, i), indices_list)

    # Sample indices with the given fitness weights
    sampled_indices = sample(
        indices_list,
        ProbabilityWeights(fitnesses),
        pop.parameters.population_size,
        replace = true,
        ordered = false,
    )

    # Sort sampled indices to avoid unnecessary memory shuffling during offspring generation
    sort!(sampled_indices)

    # Create new offspring from sampled individuals
    for i = 1:pop.parameters.population_size
        offspring!(pop, i, sampled_indices[i])
    end

    nothing
end
=#

#= Maximal fitness reproduction
function reproduce!(pop::Population)
    # Calculate fitness for all individuals in the population
    fitnesses = map(i -> fitness_exp(pop, i), indices_list)

    # Find the index of the individual with the highest fitness
    highest_fitness_index = argmax(fitnesses)

    # Update population individuals based on maximal fitness
    for i = 1:pop.parameters.population_size
        offspring!(pop, i, highest_fitness_index)
    end

    nothing
end
=#

end
