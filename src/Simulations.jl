module Simulations

export simulation

include("SimulationParameters.jl")
include("Populations.jl")
include("Exponentials.jl")
include("Utilities.jl")
include("Objectives.jl")
include("BehavEqs.jl")
include("SocialInteractions.jl")
include("IOHandler.jl")
include("Reproductions.jl")
include("Mutations.jl")

using .SimulationParameters
using .Populations
using .Exponentials
using .Utilities
using .Objectives
using .BehavEqs
using .SocialInteractions
using .IOHandler
using .Reproductions
using .Mutations

using Core.Intrinsics, StatsBase, Random, Distributions, DataFrames, Distributed


#############
# Simulation ####################################################################################################################
#############

function simulation(pop::Population)

    ############
    # Sim init #
    ############

    output_length =
        floor(Int64, pop.parameters.generations / pop.parameters.output_save_tick) *
        pop.parameters.population_size
    outputs = DataFrame(
        generation = Vector{Int64}(undef, output_length),
        individual = Vector{Int64}(undef, output_length),
        action = Vector{Float64}(undef, output_length),
        a = Vector{Float64}(undef, output_length),
        p = Vector{Float64}(undef, output_length),
        T_ext = Vector{Float64}(undef, output_length),
        T_self = Vector{Float64}(undef, output_length),
        payoff = Vector{Float64}(undef, output_length),
    )

    truncate_bounds = truncation_bounds(pop.parameters.mutation_variance, 0.99)

    ############
    # Sim Loop #
    ############

    for t in 1:pop.parameters.generations
        # Execute social interactions and calculate payoffs
        social_interactions!(pop)

        # Per-timestep counters, outputs going to disk
        if t % pop.parameters.output_save_tick == 0
            output!(outputs, t, pop)
        end

        # Reproduction function to produce new generation
        reproduce!(pop)

        # Mutation function iterates over population and mutates at chance probability μ
        if pop.parameters.mutation_rate > 0
            mutate!(pop, truncate_bounds)
        end
    end

    return outputs
end

end # module Simulations
