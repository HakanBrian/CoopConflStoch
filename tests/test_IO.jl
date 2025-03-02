using BenchmarkTools, Revise

include("../src/Main.jl")
using .MainSimulation

#############
# IO Handler ####################################################################################################################
#############

# Example usage
sweep_vars = Dict(:T => [0.5], :N => [100])
filename_full = MainSimulation.generate_filename("simulation", sweep_vars, "Full")
filename_filtered = MainSimulation.generate_filename("simulation", sweep_vars, "Filtered", time_point=10)

println(filename_full)      # Output: "simulation_T=0.5_N=100_Full.csv"
println(filename_filtered)  # Output: "simulation_T_N_Filtered_G10.csv"
