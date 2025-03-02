module Simulations

export simulation

using ..MainSimulation.Populations
import ..MainSimulation.Populations: Population, truncation_bounds

using ..MainSimulation.SocialInteractions
import ..MainSimulation.SocialInteractions: social_interactions!

using ..MainSimulation.Reproductions
import ..MainSimulation.Reproductions: reproduce!

using ..MainSimulation.Mutations
import ..MainSimulation.Mutations: mutate!

using DataFrames

function output!(outputs::DataFrame, t::Int64, pop::Population)
    N = pop.parameters.population_size

    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * N + 1
    end

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base+N-1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = fill(t, N)
    outputs.individual[output_rows] = 1:N
    outputs.action[output_rows] = pop.action
    outputs.norm[output_rows] = pop.norm
    outputs.ext_pun[output_rows] = pop.ext_pun
    outputs.int_pun_ext[output_rows] = pop.int_pun_ext
    outputs.int_pun_self[output_rows] = pop.int_pun_self
    outputs.payoff[output_rows] = pop.payoff

    nothing
end


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
        norm = Vector{Float64}(undef, output_length),
        ext_pun = Vector{Float64}(undef, output_length),
        int_pun_ext = Vector{Float64}(undef, output_length),
        int_pun_self = Vector{Float64}(undef, output_length),
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
