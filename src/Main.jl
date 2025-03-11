module MainSimulation

# Include submodules
include("structs/SimulationParameters.jl")
include("structs/Populations.jl")
include("structs/Exponentials.jl")

include("game/Utilities.jl")
include("game/Objectives.jl")
include("game/BehavEqs.jl")

include("simulation/SocialInteractions.jl")
include("simulation/Reproductions.jl")
include("simulation/Mutations.jl")
include("simulation/Simulations.jl")

include("IOHandler.jl")
include("Statistics.jl")
include("RunSimulations.jl")
include("PlotSimulations.jl")

# Load submodules
using .SimulationParameters
using .Populations
using .Exponentials

using .Utilities
using .Objectives
using .BehavEqs

using .SocialInteractions
using .Reproductions
using .Mutations
using .Simulations

using .IOHandler
using .Statistics
using .RunSimulations
using .PlotSimulations

# Export submodules
export SimulationParameters
export Populations
export Exponentials

export Utilities
export Objectives
export BehavEqs

export SocialInteractions
export Reproductions
export Mutations
export Simulations

export IOHandler
export Statistics
export RunSimulations
export PlotSimulations

# Re-export functions and types so they are available at the top level
export SimulationParameter, update_params
export run_simulation

end # module MainSimulation
