module Populations

export Population, offspring!, truncation_bounds, population_construction

using ..SimulationParameters, Distributions, Random

mutable struct Population
    parameters::SimulationParameter
    action::Vector{Float32}
    norm::Vector{Float32}
    ext_pun::Vector{Float32}
    int_pun_ext::Vector{Float32}
    int_pun_self::Vector{Float32}
    payoff::Vector{Float32}
    interactions::Vector{Int64}
    groups::Matrix{Int64}
end

function Base.copy(pop::Population)
    return Population(
        copy(getfield(pop, :parameters)),
        copy(getfield(pop, :action)),
        copy(getfield(pop, :norm)),
        copy(getfield(pop, :ext_pun)),
        copy(getfield(pop, :int_pun_ext)),
        copy(getfield(pop, :int_pun_self)),
        copy(getfield(pop, :payoff)),
        copy(getfield(pop, :interactions)),
        copy(getfield(pop, :groups)),
    )
end

function Base.copy!(old_population::Population, new_population::Population)
    copy!(getfield(old_population, :parameters), getfield(new_population, :parameters))
    copy!(getfield(old_population, :action), getfield(new_population, :action))
    copy!(getfield(old_population, :norm), getfield(new_population, :norm))
    copy!(getfield(old_population, :ext_pun), getfield(new_population, :ext_pun))
    copy!(getfield(old_population, :int_pun_ext), getfield(new_population, :int_pun_ext))
    copy!(getfield(old_population, :int_pun_self), getfield(new_population, :int_pun_self))
    copy!(getfield(old_population, :payoff), getfield(new_population, :payoff))
    copy!(getfield(old_population, :interactions), getfield(new_population, :interactions))
    copy!(getfield(old_population, :groups), getfield(new_population, :groups))

    nothing
end

function offspring!(pop::Population, offspring_index::Int64, parent_index::Int64)
    # Copy traits from parent to offspring
    pop.action[offspring_index] = pop.action[parent_index]
    pop.norm[offspring_index] = pop.norm[parent_index]
    pop.ext_pun[offspring_index] = pop.ext_pun[parent_index]
    pop.int_pun_ext[offspring_index] = pop.int_pun_ext[parent_index]
    pop.int_pun_self[offspring_index] = pop.int_pun_self[parent_index]

    # Set initial values for offspring
    pop.payoff[offspring_index] = 0.0f0
    pop.interactions[offspring_index] = 0
end

function truncation_bounds(variance::Float64, retain_proportion::Float64)
    # Calculate tail probability alpha
    alpha = 1 - retain_proportion

    # Calculate z-score corresponding to alpha/2
    z_alpha_over_2 = quantile(Normal(), 1 - alpha / 2)

    # Calculate truncation bounds
    lower_bound = -z_alpha_over_2 * √variance
    upper_bound = z_alpha_over_2 * √variance

    return (lower_bound, upper_bound)
end

function population_construction(parameters::SimulationParameter)
    trait_variance = parameters.trait_variance
    use_distribution = trait_variance != 0

    # Collect initial traits
    action0 = parameters.action0
    norm0 = parameters.norm0
    ext_pun0 = parameters.ext_pun0
    int_pun_ext0 = parameters.int_pun_ext0
    int_pun_self0 = parameters.int_pun_self0
    pop_size = parameters.population_size
    group_size = parameters.group_size

    # Initialize arrays for attributes
    actions = Vector{Float32}(undef, pop_size)
    norms = Vector{Float32}(undef, pop_size)
    ext_puns = Vector{Float32}(undef, pop_size)
    int_puns_ext = Vector{Float32}(undef, pop_size)
    int_puns_self = Vector{Float32}(undef, pop_size)
    payoffs = Vector{Float32}(undef, pop_size)
    interactions = Vector{Int64}(undef, pop_size)
    groups = Matrix{Int64}(undef, pop_size, group_size)

    # Construct distributions if necessary
    if use_distribution
        lower_bound, upper_bound = truncation_bounds(trait_variance, 0.99)
        action0_dist = truncated(
            Normal(0, trait_variance),
            lower = max(lower_bound, -action0),
            upper = upper_bound,
        )
        norm0_dist = truncated(
            Normal(0, trait_variance),
            lower = max(lower_bound, -norm0),
            upper = upper_bound,
        )
        ext_pun0_dist = truncated(
            Normal(0, trait_variance),
            lower = max(lower_bound, -ext_pun0),
            upper = upper_bound,
        )
        int_pun_ext0_dist = truncated(
            Normal(0, trait_variance),
            lower = max(lower_bound, -int_pun_ext0),
            upper = upper_bound,
        )
        int_pun_self0_dist = truncated(
            Normal(0, trait_variance),
            lower = max(lower_bound, -int_pun_self0),
            upper = upper_bound,
        )
    end

    # Create individuals
    for i in 1:pop_size
        if use_distribution
            actions[i] = action0 + rand(action0_dist)
            norms[i] = norm0 + rand(norm0_dist)
            ext_puns[i] = ext_pun0 + rand(ext_pun0_dist)
            int_puns_ext[i] = int_pun_ext0 + rand(int_pun_ext0_dist)
            int_puns_self[i] = int_pun_self0 + rand(int_pun_self0_dist)
        else
            actions[i] = action0
            norms[i] = norm0
            ext_puns[i] = ext_pun0
            int_puns_ext[i] = int_pun_ext0
            int_puns_self[i] = int_pun_self0
        end
        payoffs[i] = 0.0f0
        interactions[i] = 0
    end

    return Population(
        parameters,
        actions,
        norms,
        ext_puns,
        int_puns_ext,
        int_puns_self,
        payoffs,
        interactions,
        groups,
    )
end

end # module SimulationParameters
