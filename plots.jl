using Plots, PlotlyJS, DataFrames


#########
# Helper ########################################################################################################################
#########

include("helper.jl")
include("processing.jl")


#######
# Plot ##########################################################################################################################
#######

function plot_simulation_data_Plots(
    df::DataFrame,
    x_axis_variable::Symbol,
    xlabel_text::String,
    display_plot::Bool = false,
)
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

    plots_array = []

    for plot_var in plot_var_set
        # Initialize plot
        p = Plots.plot(legend = true)

        # Create mean and ribbons for each trait
        for trait in plot_var
            mean_col = Symbol(trait * "_mean_mean")
            std_col = Symbol(trait * "_mean_std")

            Plots.plot!(
                p,
                df[!, x_axis_variable],
                df[!, mean_col],
                ribbon = (df[!, std_col], df[!, std_col]),
                label = trait,
                color = colors[trait*" mean"],
            )
        end

        push!(plots_array, p)  # Store plot in array

        xlabel!(xlabel_text)
        ylabel!("Traits")

        if display_plot
            display("image/png", p)
        end
    end

    return plots_array
end

plot_sim_Plots(df::DataFrame; display_plot = false) =
    plot_simulation_data_Plots(df, :generation, "Generation", display_plot)

plot_sweep_r_Plots(df::DataFrame; display_plot = false) =
    plot_simulation_data_Plots(df, :relatedness, "Relatedness", display_plot)

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
    df::DataFrame,
    x_axis_variable::Symbol,
    xlabel_text::String,
    title_text::String,
)
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

    # Generate hover text dynamically
    for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
        hover_col = Symbol(trait * "_mean_hover")
        mean_col = Symbol(trait * "_mean_mean")
        std_col = Symbol(trait * "_mean_std")

        df[!, hover_col] =
            xlabel_text .* ": " .* string.(df[!, x_axis_variable]) .* "<br>" .* trait .*
            " Mean: " .* string.(df[!, mean_col]) .* "<br>Std Dev: " .*
            string.(df[!, std_col])
    end

    # Plot replicate means with ribbons for standard deviation
    for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
        mean_col = Symbol(trait * "_mean_mean")
        std_col = Symbol(trait * "_mean_std")
        hover_col = Symbol(trait * "_mean_hover")

        # Plot mean
        add_trace!(
            p,
            PlotlyJS.scatter(
                x = df[!, x_axis_variable],
                y = df[!, mean_col],
                mode = "lines",
                line_color = colors[trait],
                name = trait,
                hovertext = df[!, hover_col],
                hoverinfo = "text",
            ),
        )

        # Plot ribbons for standard deviation (upper bounds)
        add_trace!(
            p,
            PlotlyJS.scatter(
                x = df[!, x_axis_variable],
                y = df[!, mean_col] .+ df[!, std_col],
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
                x = df[!, x_axis_variable],
                y = df[!, mean_col] .- df[!, std_col],
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
        title = title_text,
        xaxis_title = xlabel_text,
        yaxis_title = "Traits",
        width = 600,
        height = 400,
        legend = Dict(:orientation => "h", :x => 0, :y => -0.2),
        hovermode = "x unified",
    )

    # Display plot
    display(p)
end

plot_sim_Plotly(df::DataFrame) =
    plot_simulation_data_Plotly(df, :generation, "Generation", "Mean of Replicates")

plot_sweep_r_Plotly(df::DataFrame) =
    plot_simulation_data_Plotly(df, :relatedness, "Relatedness", "Mean of Replicates")

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
