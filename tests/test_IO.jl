using BenchmarkTools, Revise

push!(LOAD_PATH, abspath("src"));
using MainSimulation

# Apply changes
Revise.revise()


#############
# IO Handler ####################################################################################################################
#############

# Example usage
sweep_vars = Dict(:T => [0.5], :N => [100])
filename_full = MainSimulation.generate_filename("simulation", sweep_vars, "Full")
filename_filtered =
    MainSimulation.generate_filename("simulation", sweep_vars, "Filtered", time_point = 10)

println(filename_full)      # Output: "simulation_T=0.5_N=100_Full.csv"
println(filename_filtered)  # Output: "simulation_T_N_Filtered_G10.csv"

# loading simulations
eZero = MainSimulation.read_matching_simulations(
    "data/eZero/",
    pattern_template = "eZero_{punishment}_relatedness_Filtered_G100000.csv",
    extract_keys = ["punishment"],
)

eip = MainSimulation.read_matching_simulations("data/basin/ext_int_pun/",
        pattern_template = "ext_int_pun_unipenal_group_size={group_size}_relatedness={relatedness}_Full.csv",
        extract_keys = ["group_size", "relatedness"])
