using BenchmarkTools, Revise

push!(LOAD_PATH, abspath("src"));
using MainSimulation

# Apply changes
Revise.revise()


##################
# PlotSimulations ###############################################################################################################
##################

# z_var === nothing
sims =
    MainSimulation.read_matching_simulations("data/default/", "default_*_relatedness_*.csv")

MainSimulation.plot_simulation_Plots(
    sims["bipenal"],
    :relatedness,
    "relatedness",
    display_plot = false,
)

plots_dict =
    MainSimulation.plot_multiple_simulations_Plots(sims, :relatedness, "relatedness")

plots_list = MainSimulation.extract_plot_lists(plots_dict)

MainSimulation.compare_plot_lists(plots_list)


# z_var !== nothing
sims =
    MainSimulation.read_matching_simulations("data/rDiffGS/", "rDiffGS_*_relatedness_*.csv")

MainSimulation.plot_simulation_Plots(
    sims["bipenal_group_size"],
    :relatedness,
    "relatedness",
    z_var = :group_size,
    display_plot = false,
)

plots_dict = MainSimulation.plot_multiple_simulations_Plots(
    sims,
    :relatedness,
    "relatedness",
    z_var = :group_size,
)

plots_list = MainSimulation.extract_plot_lists(plots_dict)

MainSimulation.compare_plot_lists(plots_list)


# Heatmap smoothing issue
using Plots, PlotlyJS
plotlyjs()
gr()
p = Plots.heatmap(rand(24, 24))
Plots.pdf(p, "heatmap_plotly")