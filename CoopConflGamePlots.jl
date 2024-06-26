using Plots


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Population Construction
##################

my_parameter = simulation_parameters(0.2, 0.5, 0.4, 0.0, 100000, 5, 500, 0.0, 0.5, 0.0, 0.0005, 10)

my_population = population_construction(my_parameter);


##################
# Simulation
##################

@time my_simulation = simulation(my_population)


##################
# Plot
##################

# Plotting each of generations's average data
my_simulation_gdf = groupby(my_simulation, :generation);
my_simulation_mean = combine(my_simulation_gdf, :action => mean, :a => mean, :p => mean, :T => mean, :payoff => mean);
plot(my_simulation_mean.generation,
    [my_simulation_mean.action_mean, my_simulation_mean.a_mean, my_simulation_mean.p_mean, my_simulation_mean.T_mean, my_simulation_mean.payoff_mean],
    title=string(my_parameter),
    titlefontsize=8,
    xlabel="Generation", ylabel="Traits",
    label=["action" "a" "p" "T" "payoff"])

# Plotting each individual's data
plot()
for i in 1:my_population.parameters.N
    individual_data = filter(row -> row[:individual] == 10, my_simulation)
    plot!(individual_data.generation, individual_data.action, label="", linestyle=:solid, color=:blue)
    plot!(individual_data.generation, individual_data.a, label="", linestyle=:dash, color=:red)
    plot!(individual_data.generation, individual_data.p, label="", linestyle=:dot, color=:green)
    plot!(individual_data.generation, individual_data.T, label="", linestyle=:dashdot, color=:magenta)
    plot!(individual_data.generation, individual_data.payoff, label="", linestyle=:dashdotdot, color=:brown)
end

# Display the plot with appropriate labels and title
plot!(title=string(my_parameter),
      xlabel="Generation", ylabel="Traits", legend=:outertopright, titlefontsize=8)

display(plot)