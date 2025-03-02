module PlotSimulations

export plot_sim_Plots,
    plot_sweep_r_Plots,
    plot_sweep_rep_Plots,
    plot_sweep_rip_Plots,
    plot_sweep_rgs_Plots,
    compare_plot_lists,
    plot_sim_Plotly,
    plot_sweep_r_Plotly,
    plot_sweep_rep_Plotly,
    plot_sweep_rip_Plotly,
    plot_sweep_rgs_Plotly

using Plots, PlotlyJS, DataFrames


########
# Plots #########################################################################################################################
########

function plot_simulation_data_Plots(
    df::DataFrame,
    x_axis_variable::Symbol,
    xlabel_text::String,
    display_plot::Bool = false,
)
    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "norm" => :red,
        "ext_pun" => :green,
        "int_pun_ext" => :purple,
        "int_pun_self" => :yellow,
        "payoff" => :orange,
        "action mean" => :blue4,
        "norm mean" => :red4,
        "ext_pun mean" => :green4,
        "int_pun_ext mean" => :purple4,
        "int_pun_self mean" => :yellow4,
        "payoff mean" => :orange4,
    )

    # Define dependent variables to plot
    plot_var_set = [
        ["action", "norm", "ext_pun", "int_pun_ext", "int_pun_self", "payoff"],
        ["ext_pun", "int_pun_ext", "int_pun_self"],
        ["ext_pun"],
        ["action", "norm"],
    ]

    plots_array = []

    for plot_var in plot_var_set
        # Initialize plot
        p = Plots.plot(legend = true, fmt = :pdf)

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
            display(p)
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
            fmt = :pdf,
        )

        push!(plots_array, p)  # Store plot in array

        # Conditionally display plot
        if display_plot
            display(p)
        end
    end

    return plots_array  # Return all plots
end

function plot_sweep_rep_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plots(
        statistics,
        :relatedness,
        :ext_pun0,
        dependent_vars,
        display_plot = display_plot,
    )
end

function plot_sweep_rip_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [:action_mean_mean, :norm_mean_mean, :ext_pun_mean_mean, :payoff_mean_mean]
    plot_sweep_heatmap_Plots(
        statistics,
        :relatedness,
        :int_pun_ext0,
        dependent_vars,
        display_plot = display_plot,
    )
end

function plot_sweep_rgs_Plots(statistics::DataFrame; display_plot::Bool = false)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :ext_pun_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
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


################
# Compare Plots #################################################################################################################
################

function compare_plot_lists(plot_lists::Vector{Vector{Any}})
    num_sets = length(plot_lists)  # Number of sets of plots
    num_plots = length(plot_lists[1])  # Number of plots per set

    # Ensure all plot lists have the same number of plots
    @assert all(length(p) == num_plots for p in plot_lists) "All plot lists must have the same number of plots!"

    for i in 1:num_plots
        # Gather all corresponding plots from each list
        plots_i = [plots[i] for plots in plot_lists]

        # Determine global limits for x, y, and z across all plots in this index
        xlims_global = (
            minimum(Plots.xlims(p)[1] for p in plots_i),
            maximum(Plots.xlims(p)[2] for p in plots_i),
        )
        ylims_global = (
            minimum(Plots.ylims(p)[1] for p in plots_i),
            maximum(Plots.ylims(p)[2] for p in plots_i),
        )
        clims_global = (
            minimum(Plots.zlims(p)[1] for p in plots_i),
            maximum(Plots.zlims(p)[2] for p in plots_i),
        )

        # Create a grid layout based on the number of plot sets
        p = Plots.plot(
            plots_i...;
            layout = (1, num_sets),
            size = (600 * num_sets, 400),
            xlims = xlims_global,
            ylims = ylims_global,
            clims = clims_global,
            fmt = :pdf,
        )

        display(p)
    end
end


###########
# PlotlyJS ######################################################################################################################
###########

function plot_simulation_data_Plotly(
    df::DataFrame,
    x_axis_variable::Symbol,
    xlabel_text::String,
    title_text::String,
)
    # Initialize plot
    p = PlotlyJS.Plot()

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "norm" => :red,
        "ext_pun" => :green,
        "int_pun_ext" => :purple,
        "int_pun_self" => :yellow,
        "payoff" => :orange,
        "action_stdev" => "rgba(0,0,255,0.2)",
        "norm_stdev" => "rgba(255,0,0,0.2)",
        "ext_pun_stdev" => "rgba(0,255,0,0.2)",
        "int_pun_ext_stdev" => "rgba(128,0,128,0.2)",
        "int_pun_self_stdev" => "rgba(255,255,0,0.2)",
        "payoff_stdev" => "rgba(255,165,0,0.2)",
    )

    # Generate hover text dynamically
    for trait in ["action", "norm", "ext_pun", "int_pun_ext", "int_pun_self", "payoff"]
        hover_col = Symbol(trait * "_mean_hover")
        mean_col = Symbol(trait * "_mean_mean")
        std_col = Symbol(trait * "_mean_std")

        df[!, hover_col] =
            xlabel_text .* ": " .* string.(df[!, x_axis_variable]) .* "<br>" .* trait .*
            " Mean: " .* string.(df[!, mean_col]) .* "<br>Std Dev: " .*
            string.(df[!, std_col])
    end

    # Plot replicate means with ribbons for standard deviation
    for trait in ["action", "norm", "ext_pun", "int_pun_ext", "int_pun_self", "payoff"]
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

        # Display plot
        display(p)
    end
end

function plot_sweep_rep_Plotly(statistics::DataFrame)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :ext_pun0, dependent_vars)
end

function plot_sweep_rip_Plotly(statistics::DataFrame)
    dependent_vars = [:action_mean_mean, :norm_mean_mean, :ext_pun_mean_mean, :payoff_mean_mean]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :int_pun_ext0, dependent_vars)
end

function plot_sweep_rgs_Plotly(statistics::DataFrame)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :ext_pun_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
        :payoff_mean_mean,
    ]
    plot_sweep_heatmap_Plotly(statistics, :relatedness, :group_size, dependent_vars)
end

end # module Plots
