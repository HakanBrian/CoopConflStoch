module Populations

export Population

using ..SimulationParameters

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

end # module SimulationParameters