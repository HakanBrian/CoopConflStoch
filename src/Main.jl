module MainSimulation


include("structs/SimulationParameters.jl")
using .SimulationParameters

include("structs/Populations.jl")
using .Populations

include("structs/Exponentials.jl")
using .Exponentials

include("game/Utilities.jl")
using .Utilities

include("game/Objectives.jl")
using .Objectives

include("game/BehavEqs.jl")
using .BehavEqs

include("simulation/SocialInteractions.jl")
using .SocialInteractions

include("simulation/Reproductions.jl")
using .Reproductions

include("simulation/Mutations.jl")
using .Mutations

include("simulation/Simulations.jl")
using .Simulations

include("IOHandler.jl")
using .IOHandler

include("Statistics.jl")
using .Statistics

include("RunSimulations.jl")
using .RunSimulations

include("PlotSimulations.jl")
using .PlotSimulations


end # module Main
