using Plots, PlotlyJS, DataFrames


#########
# Helper ########################################################################################################################
#########

include("helper.jl")


#######
# Plot ##########################################################################################################################
#######

function plot_simulation_data_Plots(
    all_simulation_means::DataFrame;
    param_id::Union{Nothing,Int64} = nothing,
)
    # Filter the data if param_id is provided
    if param_id !== nothing
        all_simulation_means = filter(row -> row.param_id == param_id, all_simulation_means)
    end

    # Determine the number of params and replicates
    num_params = maximum(all_simulation_means.param_id)  # Assuming replicate param are consistent

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action mean" => :blue4,
        "a mean" => :red4,
        "p mean" => :green4,
        "T_ext mean" => :purple4,
        "T_self mean" => :yellow4,
        "payoff mean" => :orange4,
    )

    # Initialize the plot
    p = Plots.plot(legend = false)

    # Plot mean and ribbons for param_id's data
    for i in 1:num_params
        # Filter by param_id
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Plot mean and ribbons for each trait with a distinct label for each parameter set
        for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
            Plots.plot!(
                p,
                statistics.generation,
                statistics[!, trait*"_mean_mean"],
                ribbon = (
                    statistics[!, trait*"_mean_std"],
                    statistics[!, trait*"_mean_std"],
                ),
                label = trait * " ($i)",
                color = colors[trait*" mean"],
            )
        end
    end

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display("image/png", p)
end

function plot_sim_Plots(all_simulation_means::DataFrame)
    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action mean" => :blue4,
        "a mean" => :red4,
        "p mean" => :green4,
        "T_ext mean" => :purple4,
        "T_self mean" => :yellow4,
        "payoff mean" => :orange4,
    )

    # Initialize the plot
    p = Plots.plot(legend = true)

    # Plot mean and ribbons for each trait with a distinct label for each parameter set
    for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
        Plots.plot!(
            p,
            all_simulation_means.generation,
            all_simulation_means[!, trait*"_mean_mean"],
            ribbon = (
                all_simulation_means[!, trait*"_mean_std"],
                all_simulation_means[!, trait*"_mean_std"],
            ),
            label = trait,
            color = colors[trait*" mean"],
        )
    end

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display("image/png", p)
end

function plot_sweep_r_Plots(statistics::DataFrame; display_plot::Bool = false)
    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action mean" => :blue4,
        "a mean" => :red4,
        "p mean" => :green4,
        "T_ext mean" => :purple4,
        "T_self mean" => :yellow4,
        "payoff mean" => :orange4,
    )

    # Define dependent variables to plot
    plot_var_set = [
        ["action", "a", "p", "T_ext", "T_self", "payoff"],
        ["p", "T_ext", "T_self"],
        ["p"],
        ["action", "a"],
    ]

    # Store plots in an array
    plots_array = []

    # Plot each set of dependent variables
    for plot_var in plot_var_set
        # Initialize plot
        p = Plots.plot(legend = true)

        # Create mean and ribbons for each trait
        for trait in plot_var
            Plots.plot!(
                p,
                statistics.relatedness,
                statistics[!, trait*"_mean_mean"],
                ribbon = (
                    statistics[!, trait*"_mean_std"],
                    statistics[!, trait*"_mean_std"],
                ),
                label = trait,
                color = colors[trait*" mean"],
            )
        end

        push!(plots_array, p)  # Store plot in array

        # Display plot
        xlabel!("Relatedness")
        ylabel!("Value")

        # Conditionally display plot
        if display_plot
            display("image/png", p)
        end
    end

    return plots_array  # Return all plots
end

function plot_sweep_heatmap_Plots(
    statistics::DataFrame,
    x_var::Symbol,
    y_var::Symbol,
    dependent_vars::Vector{Symbol};
    display_plot::Bool = false,
)
    # Get unique sorted values for x and y axes
    x_values = sort(unique(statistics[!, x_var]))
    y_values = sort(unique(statistics[!, y_var]))

    # Store plots in an array
    plots_array = []

    # Plot each dependent variable as a separate heatmap
    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, y_var, x_var, var)

        # Convert DataFrame to a matrix (remove `y_var` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(y_var)])

        # Create heatmap plot
        p = Plots.heatmap(
            x_values,
            y_values,
            heatmap_matrix,
            color = :viridis,
            xlabel = string(x_var),
            ylabel = string(y_var),
            title = "Heatmap of $var",
            colorbar_title = "Value",
        )

        push!(plots_array, p)  # Store plot in array

        # Conditionally display plot
        if display_plot
            display("image/png", p)
        end
    end

    return plots_array  # Return all plots
end

function plot_sweep_rep_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [
        :action_mean_mean,
        :a_mean_mean,
        :T_ext_mean_mean,
        :T_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plots(
        statistics,
        :relatedness,
        :ext_pun,
        dependent_vars,
        display_plot = display_plot,
    )
end

function plot_sweep_rip_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :payoff_mean_mean]
    plot_sweep_heatmap_Plots(
        statistics,
        :relatedness,
        :int_pun,
        dependent_vars,
        display_plot = display_plot,
    )
end

function plot_sweep_rgs_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [
        :action_mean_mean,
        :a_mean_mean,
        :p_mean_mean,
        :T_ext_mean_mean,
        :T_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plots(
        statistics,
        :relatedness,
        :group_size,
        dependent_vars,
        display_plot = display_plot,
    )
end


##########
# Compare #######################################################################################################################
##########

function compare_plot_lists(plots1::Vector{Any}, plots2::Vector{Any})
    @assert length(plots1) == length(plots2) "Both lists must have the same number of plots!"

    for i in 1:length(plots1)
        # Determine x, y and z limits for this specific pair
        xlims_pair = (
            minimum([Plots.xlims(plots1[i])[1], Plots.xlims(plots2[i])[1]]),
            maximum([Plots.xlims(plots1[i])[2], Plots.xlims(plots2[i])[2]]),
        )
        ylims_pair = (
            minimum([Plots.ylims(plots1[i])[1], Plots.ylims(plots2[i])[1]]),
            maximum([Plots.ylims(plots1[i])[2], Plots.ylims(plots2[i])[2]]),
        )
        clims_pair = (
            minimum([Plots.zlims(plots1[i])[1], Plots.zlims(plots2[i])[1]]),
            maximum([Plots.zlims(plots1[i])[2], Plots.zlims(plots2[i])[2]]),
        )

        # Display the comparison with synchronized limits for this pair
        p = Plots.plot(
            plots1[i],
            plots2[i];
            layout = (1, 2),
            size = (1200, 400),
            xlims = xlims_pair,
            ylims = ylims_pair,
            clims = clims_pair,
        )

        display(p)
    end
end


#########
# Plotly ########################################################################################################################
#########

function plot_simulation_data_Plotly(
    all_simulation_means::DataFrame;
    param_id::Union{Nothing,Int64} = nothing,
)
    # Filter the data if param_id is provided
    if param_id !== nothing
        all_simulation_means = filter(row -> row.param_id == param_id, all_simulation_means)
    end

    # Determine the number of params and replicates
    num_params = maximum(all_simulation_means.param_id)  # Assuming param numbers are consistent

    # Initialize plot
    p_means = Plot()

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action_stdev" => "rgba(0,0,255,0.2)",
        "a_stdev" => "rgba(255,0,0,0.2)",
        "p_stdev" => "rgba(0,255,0,0.2)",
        "T_ext_stdev" => "rgba(128,0,128,0.2)",
        "T_self_stdev" => "rgba(255,255,0,0.2)",
        "payoff_stdev" => "rgba(255,165,0,0.2)",
    )

    # Plot each param_id's data
    for i in 1:num_params
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current param_id
        statistics = calculate_statistics(param_data)

        # Create formatted hover text for each trait
        statistics[!, :action_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>action Mean: " .*
            string.(statistics.action_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.action_mean_std)
        statistics[!, :a_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>a Mean: " .*
            string.(statistics.a_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.a_mean_std)
        statistics[!, :p_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>p Mean: " .*
            string.(statistics.p_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.p_mean_std)
        statistics[!, :T_ext_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>T_ext Mean: " .*
            string.(statistics.T_ext_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.T_ext_mean_std)
        statistics[!, :T_self_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>T_self Mean: " .*
            string.(statistics.T_self_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.T_self_mean_std)
        statistics[!, :payoff_mean_hover] =
            "Generation: " .* string.(statistics.generation) .* "<br>payoff Mean: " .*
            string.(statistics.payoff_mean_mean) .* "<br>Std Dev: " .*
            string.(statistics.payoff_mean_std)

        # Plot replicate means with ribbons for standard deviation
        for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
            # Plot replicate means
            add_trace!(
                p_means,
                PlotlyJS.scatter(
                    x = statistics.generation,
                    y = statistics[!, trait*"_mean_mean"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = trait * " ($i)",
                    hovertext = statistics[!, trait*"_mean_hover"],
                    hoverinfo = "text",
                ),
            )

            # Plot ribbons for standard deviation (upper bounds)
            add_trace!(
                p_means,
                PlotlyJS.scatter(
                    x = statistics.generation,
                    y = statistics[!, trait*"_mean_mean"] .+
                        statistics[!, trait*"_mean_std"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = "",
                    fill = "tonexty",
                    fillcolor = colors[trait*"_stdev"],
                    line = Dict(:width => 0),
                    hoverinfo = "none",
                    showlegend = false,
                ),
            )

            # Plot ribbons for standard deviation (lower bounds)
            add_trace!(
                p_means,
                PlotlyJS.scatter(
                    x = statistics.generation,
                    y = statistics[!, trait*"_mean_mean"] .-
                        statistics[!, trait*"_mean_std"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = "",
                    fill = "tonexty",
                    fillcolor = colors[trait*"_stdev"],
                    line = Dict(:width => 0),
                    hoverinfo = "none",
                    showlegend = false,
                ),
            )
        end
    end

    # Layout for replicate means
    relayout!(
        p_means,
        title = "Mean of Replicates",
        xaxis_title = "Generation",
        yaxis_title = "Traits",
        width = 600,   # Set width to 600px
        height = 400,   # Set height to 400px
        legend = Dict(:orientation => "h", :x => 0, :y => -0.2),
        hovermode = "x unified",
    )

    # Display plots
    display(p_means)
end

function plot_sim_Plotly(all_simulation_means::DataFrame)
    # Initialize plot
    p = Plot()

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action_stdev" => "rgba(0,0,255,0.2)",
        "a_stdev" => "rgba(255,0,0,0.2)",
        "p_stdev" => "rgba(0,255,0,0.2)",
        "T_ext_stdev" => "rgba(128,0,128,0.2)",
        "T_self_stdev" => "rgba(255,255,0,0.2)",
        "payoff_stdev" => "rgba(255,165,0,0.2)",
    )

    # Create formatted hover text for each trait
    all_simulation_means[!, :action_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>action Mean: " .*
        string.(all_simulation_means.action_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.action_mean_std)
    all_simulation_means[!, :a_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>a Mean: " .*
        string.(all_simulation_means.a_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.a_mean_std)
    all_simulation_means[!, :p_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>p Mean: " .*
        string.(all_simulation_means.p_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.p_mean_std)
    all_simulation_means[!, :T_ext_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>T_ext Mean: " .*
        string.(all_simulation_means.T_ext_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.T_ext_mean_std)
    all_simulation_means[!, :T_self_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>T_self Mean: " .*
        string.(all_simulation_means.T_self_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.T_self_mean_std)
    all_simulation_means[!, :payoff_mean_hover] =
        "Generation: " .* string.(all_simulation_means.generation) .* "<br>payoff Mean: " .*
        string.(all_simulation_means.payoff_mean_mean) .* "<br>Std Dev: " .*
        string.(all_simulation_means.payoff_mean_std)

    # Plot replicate means with ribbons for standard deviation
    for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
        # Plot replicate means
        add_trace!(
            p,
            PlotlyJS.scatter(
                x = all_simulation_means.generation,
                y = all_simulation_means[!, trait*"_mean_mean"],
                mode = "lines",
                line_color = colors[trait],
                name = trait * " ($i)",
                hovertext = all_simulation_means[!, trait*"_mean_hover"],
                hoverinfo = "text",
            ),
        )

        # Plot ribbons for standard deviation (upper bounds)
        add_trace!(
            p,
            PlotlyJS.scatter(
                x = all_simulation_means.generation,
                y = all_simulation_means[!, trait*"_mean_mean"] .+
                    all_simulation_means[!, trait*"_mean_std"],
                mode = "lines",
                line_color = colors[trait],
                name = "",
                fill = "tonexty",
                fillcolor = colors[trait*"_stdev"],
                line = Dict(:width => 0),
                hoverinfo = "none",
                showlegend = false,
            ),
        )

        # Plot ribbons for standard deviation (lower bounds)
        add_trace!(
            p,
            PlotlyJS.scatter(
                x = all_simulation_means.generation,
                y = all_simulation_means[!, trait*"_mean_mean"] .-
                    all_simulation_means[!, trait*"_mean_std"],
                mode = "lines",
                line_color = colors[trait],
                name = "",
                fill = "tonexty",
                fillcolor = colors[trait*"_stdev"],
                line = Dict(:width => 0),
                hoverinfo = "none",
                showlegend = false,
            ),
        )
    end

    # Layout for replicate means
    relayout!(
        p,
        title = "Mean of Replicates",
        xaxis_title = "Generation",
        yaxis_title = "Traits",
        width = 600,   # Set width to 600px
        height = 400,   # Set height to 400px
        legend = Dict(:orientation => "h", :x => 0, :y => -0.2),
        hovermode = "x unified",
    )

    # Display plots
    display(p)
end

function plot_sweep_r_Plotly(statistics::DataFrame)
    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T_ext" => :purple,
        "T_self" => :yellow,
        "payoff" => :orange,
        "action_stdev" => "rgba(0,0,255,0.2)",
        "a_stdev" => "rgba(255,0,0,0.2)",
        "p_stdev" => "rgba(0,255,0,0.2)",
        "T_ext_stdev" => "rgba(128,0,128,0.2)",
        "T_self_stdev" => "rgba(255,255,0,0.2)",
        "payoff_stdev" => "rgba(255,165,0,0.2)",
    )

    # Create formatted hover text for each trait
    statistics[!, :action_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>action Mean: " .*
        string.(statistics.action_mean_mean) .* "<br>Std Dev: " .*
        string.(statistics.action_mean_std)
    statistics[!, :a_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>a Mean: " .*
        string.(statistics.a_mean_mean) .* "<br>Std Dev: " .* string.(statistics.a_mean_std)
    statistics[!, :p_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>p Mean: " .*
        string.(statistics.p_mean_mean) .* "<br>Std Dev: " .* string.(statistics.p_mean_std)
    statistics[!, :T_ext_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>T_ext Mean: " .*
        string.(statistics.T_ext_mean_mean) .* "<br>Std Dev: " .*
        string.(statistics.T_ext_mean_std)
    statistics[!, :T_self_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>T_self Mean: " .*
        string.(statistics.T_self_mean_mean) .* "<br>Std Dev: " .*
        string.(statistics.T_self_mean_std)
    statistics[!, :payoff_mean_hover] =
        "Relatedness: " .* string.(statistics.relatedness) .* "<br>payoff Mean: " .*
        string.(statistics.payoff_mean_mean) .* "<br>Std Dev: " .*
        string.(statistics.payoff_mean_std)

    # Define dependent variables to plot    
    plot_var_set = [
        ["action", "a", "p", "T_ext", "T_self", "payoff"],
        ["p", "T_ext", "T_self"],
        ["p"],
        ["action", "a"],
    ]

    # Plot each set of dependent variables
    for plot_var in plot_var_set
        # Initialize plot
        p = Plot()

        # Plot replicate means with ribbons for standard deviation
        for trait in plot_var
            # Plot replicate means
            add_trace!(
                p,
                PlotlyJS.scatter(
                    x = statistics.relatedness,
                    y = statistics[!, trait*"_mean_mean"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = trait,
                    hovertext = statistics[!, trait*"_mean_hover"],
                    hoverinfo = "text",
                ),
            )

            # Plot ribbons for standard deviation (uppder bounds)
            add_trace!(
                p,
                PlotlyJS.scatter(
                    x = statistics.relatedness,
                    y = statistics[!, trait*"_mean_mean"] .+
                        statistics[!, trait*"_mean_std"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = "",
                    fill = "tonexty",
                    fillcolor = colors[trait*"_stdev"],
                    line = Dict(:width => 0),
                    hoverinfo = "none",
                    showlegend = false,
                ),
            )

            # Plot ribbons for standard deviation (lower bound)
            add_trace!(
                p,
                PlotlyJS.scatter(
                    x = statistics.relatedness,
                    y = statistics[!, trait*"_mean_mean"] .-
                        statistics[!, trait*"_mean_std"],
                    mode = "lines",
                    line_color = colors[trait],
                    name = "",
                    fill = "tonexty",
                    fillcolor = colors[trait*"_stdev"],
                    line = Dict(:width => 0),
                    hoverinfo = "none",
                    showlegend = false,
                ),
            )
        end

        # Layout for replicate means
        relayout!(
            p,
            title = "Mean of Replicates",
            xaxis_title = "Relatedness",
            yaxis_title = "Traits",
            width = 600,   # Set width to 600px
            height = 400,   # Set height to 400px
            legend = Dict(:orientation => "h", :x => 0, :y => -0.2),
            hovermode = "x unified",
        )

        # Display plots
        display(p)
    end
end

function plot_sweep_heatmap_Plotly(
    statistics::DataFrame,
    x_var::Symbol,
    y_var::Symbol,
    dependent_vars::Vector{Symbol},
)
    # Get unique sorted values for x and y axes
    x_values = sort(unique(statistics[!, x_var]))
    y_values = sort(unique(statistics[!, y_var]))

    # Store plots in an array
    plots_array = []

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, y_var, x_var, var)

        # Convert DataFrame to a matrix (remove `y_var` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(y_var)])

        # Create a heatmap trace
        trace = PlotlyJS.heatmap(
            z = heatmap_matrix,  # Data matrix
            x = x_values,  # X-axis values
            y = y_values,  # Y-axis values
            colorscale = "Viridis",
            colorbar_title = "Value",
        )

        # Define layout
        layout = Layout(
            title = "Heatmap of $var",
            xaxis_title = string(x_var),
            yaxis_title = string(y_var),
            width = 600,   # Set width to 600px
            height = 400,   # Set height to 400px
        )

        # Create the plot
        p = PlotlyJS.plot([trace], layout)
        push!(plots_array, p)

        # Display plot
        display(p)
    end

    return plots_array
end

function plot_sweep_rep_Plotly(statistics::DataFrame)
    dependent_vars = [
        :action_mean_mean,
        :a_mean_mean,
        :T_ext_mean_mean,
        :T_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :ext_pun, dependent_vars)
end

function plot_sweep_rip_Plotly(statistics::DataFrame)
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :payoff_mean_mean]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :int_pun, dependent_vars)
end

function plot_sweep_rgs_Plotly(statistics::DataFrame)
    dependent_vars = [
        :action_mean_mean,
        :a_mean_mean,
        :p_mean_mean,
        :T_ext_mean_mean,
        :T_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :group_size, dependent_vars)
end
