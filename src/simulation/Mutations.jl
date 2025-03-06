module Mutations

export mutate!

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population

using Distributions, StatsBase, Random

function mutate!(pop::Population, truncate_bounds::Tuple{Float64,Float64})
    mutation_variance = pop.parameters.mutation_variance

    # Return immediately if no mutation is needed
    if mutation_variance == 0
        return nothing
    end

    mutation_rate = pop.parameters.mutation_rate
    lower_bound, upper_bound = truncate_bounds
    mutation_dist = Normal(0, mutation_variance)

    # Define distributions for mutation
    for i in 1:pop.parameters.population_size
        # Mutate `norm` trait
        if pop.parameters.norm_mutation_enabled && rand() <= mutation_rate
            norm_dist = truncated(
                mutation_dist,
                lower = max(lower_bound, -pop.norm[i]),
                upper = upper_bound,
            )
            pop.norm[i] += rand(norm_dist)
        end

        # Mutate `ext_pun` trait
        if pop.parameters.ext_pun_mutation_enabled && rand() <= mutation_rate
            ext_pun_dist = truncated(
                mutation_dist,
                lower = max(lower_bound, -pop.ext_pun[i]),
                upper = upper_bound,
            )
            pop.ext_pun[i] += rand(ext_pun_dist)
        end

        # Mutate `int_pun_ext` trait
        if pop.parameters.int_pun_ext_mutation_enabled && rand() <= mutation_rate
            int_pun_ext_dist = truncated(
                mutation_dist,
                lower = max(lower_bound, -pop.int_pun_ext[i]),
                upper = upper_bound,
            )
            pop.int_pun_ext[i] += rand(int_pun_ext_dist)
        end

        # Mutate `int_pun_self` trait
        if pop.parameters.int_pun_self_mutation_enabled && rand() <= mutation_rate
            int_pun_self_dist = truncated(
                mutation_dist,
                lower = max(lower_bound, -pop.int_pun_self[i]),
                upper = upper_bound,
            )
            pop.int_pun_self[i] += rand(int_pun_self_dist)
        end
    end

    nothing
end

#= Mutation units
function mutate!(pop::Population, truncate_bounds::SArray{Tuple{2},Float64})
    mutation_unit = pop.parameters.mutation_variance

    # Only mutate if necessary
    if mutation_unit == 0
        return nothing
    end

    mutation_direction = [-1, 1]
    mutation_rate = pop.parameters.mutation_rate

    # Iterate over each individual by index
    for i = 1:pop.parameters.population_size
        # Mutate `norm` trait
        if pop.parameters.norm_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.norm[i] = max(0, pop.norm[i] + mutation_amount)
        end

        # Mutate `ext_pun` trait
        if pop.parameters.ext_pun_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.ext_pun[i] = max(0, pop.ext_pun[i] + mutation_amount)
        end

        # Mutate `int_pun_ext` trait
        if pop.parameters.int_pun_ext_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_ext[i] = max(0, pop.int_pun_ext[i] + mutation_amount)
        end

        # Mutate `int_pun_self` trait
        if pop.parameters.int_pun_self_mutation_enabled && rand() <= mutation_rate
            mutation_amount = rand(mutation_direction) * mutation_unit
            pop.int_pun_self[i] = max(0, pop.int_pun_self[i] + mutation_amount)
        end
    end

    nothing
end
=#

end # module Mutations
