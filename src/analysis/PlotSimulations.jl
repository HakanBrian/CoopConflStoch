module PlotSimulations

export plot_simulation_Plots,
    plot_multiple_simulations_Plots,
    plot_sweep_rep_Plots,
    plot_sweep_rip_Plots,
    plot_sweep_rgs_Plots,
    extract_plot_lists,
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

function plot_simulation_Plots(
    df::DataFrame,
    x_var::Symbol;
    dataset_name::String = "Simulation Data",
    z_var::Union{Symbol,Nothing} = nothing,
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

    # Determine unique z values (or use a single group if z_var is nothing)
    z_values = z_var !== nothing ? sort(unique(df[!, z_var])) : [nothing]

    # Storage for plots
    plots_list = []

    for z_val in z_values
        # Filter dataframe if z_var is provided
        df_subset = z_var !== nothing ? filter(row -> row[z_var] == z_val, df) : df

        for plot_var in plot_var_set
            # Construct the title with dataset name
            title_text = if z_var !== nothing
                "$dataset_name | $z_var = $z_val"
            else
                "$dataset_name"
            end

            p = Plots.plot(title = title_text, legend = true, fmt = :pdf)

            # Create mean and ribbons for each trait
            for trait in plot_var
                mean_col = Symbol(trait * "_mean_mean")
                std_col = Symbol(trait * "_mean_std")

                Plots.plot!(
                    p,
                    df_subset[!, x_var],
                    df_subset[!, mean_col],
                    ribbon = (df_subset[!, std_col], df_subset[!, std_col]),
                    label = trait,
                    color = colors[trait*" mean"],
                )
            end

            xlabel!(p, "$x_var")
            ylabel!(p, "Traits")

            push!(plots_list, p)

            if display_plot
                display(p)
            end
        end
    end

    # If z_var is nothing, return a flat vector of Plots.Plot instead of a nested vector
    return z_var === nothing ? plots_list :
           collect(Iterators.partition(plots_list, length(plot_var_set)))
end

function plot_multiple_simulations_Plots(
    dfs::Dict{String,DataFrame},
    x_var::Symbol;
    z_var::Union{Symbol,Nothing} = nothing,
)
    # Dictionary to store results
    z_var === nothing ? results = Dict{String,Vector{Plots.Plot}}() :
    results = Dict{String,Vector{Vector{Plots.Plot}}}()

    for (key, df) in dfs
        println("Processing dataset: ", key)

        # Pass dataset name (string key) to plot_simulation_data_Plots
        plots = plot_simulation_Plots(
            df,
            x_var;
            dataset_name = key,  # Key is now part of the plot title
            z_var = z_var,
            display_plot = false,
        )

        # Store results: a flat vector if z_var is nothing, else a nested vector
        results[key] = plots
    end

    return results
end

function plot_sweep_heatmap_Plots(
    df::DataFrame,
    x_var::Symbol,
    y_var::Symbol,
    dependent_vars::Vector{Symbol};
    dataset_name::String = "Simulation Data",
    z_var::Union{Symbol,Nothing} = nothing,
    display_plot::Bool = false,
)
    # Define color scheme
    colormap = :viridis

    # Determine unique z values (or use a single group if z_var is nothing)
    z_values = z_var !== nothing ? sort(unique(df[!, z_var])) : [nothing]

    # Storage for plots
    plots_list = []

    for z_val in z_values
        # Filter dataframe if z_var is provided
        df_subset = z_var !== nothing ? filter(row -> row[z_var] == z_val, df) : df

        # Get unique sorted values for x and y axes
        x_values = sort(unique(df_subset[!, x_var]))
        y_values = sort(unique(df_subset[!, y_var]))

        for var in dependent_vars
            # Pivot the data for the current dependent variable
            heatmap_data = unstack(df_subset, y_var, x_var, var)

            # Convert DataFrame to a matrix (remove `y_var` column)
            heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(y_var)])

            # Construct the title
            title_text = if z_var !== nothing
                "$dataset_name | $z_var = $z_val"
            else
                "$dataset_name"
            end

            # Create heatmap plot
            p = Plots.heatmap(
                x_values,
                y_values,
                heatmap_matrix,
                color = colormap,
                xlabel = string(x_var),
                ylabel = string(y_var),
                title = title_text,
                colorbar_title = string(var),
                fmt = :pdf,
            )

            push!(plots_list, p)  # Store plot in array

            # Conditionally display plot
            if display_plot
                display(p)
            end
        end
    end

    # If z_var is nothing, return a flat vector of Plots.Plot instead of a nested vector
    return z_var === nothing ? plots_list :
           collect(Iterators.partition(plots_list, length(dependent_vars)))
end

function plot_multiple_sweep_heatmap_Plots(
    dfs::Dict{String,DataFrame},
    x_var::Symbol,
    y_var::Symbol,
    dependent_vars::Vector{Symbol};
    z_var::Union{Symbol,Nothing} = nothing,
)
    # Dictionary to store results
    z_var === nothing ? results = Dict{String,Vector{Plots.Plot}}() :
    results = Dict{String,Vector{Vector{Plots.Plot}}}()

    for (key, df) in dfs
        println("Processing dataset: ", key)

        # Pass dataset name (string key) to plot_simulation_data_Plots
        plots = plot_sweep_heatmap_Plots(
            df,
            x_var,
            y_var,
            dependent_vars;
            dataset_name = key,
            z_var = z_var,
            display_plot = false,
        )

        # Store results: a flat vector if z_var is nothing, else a nested vector
        results[key] = plots
    end

    return results
end

function plot_sweep_rep_Plots(
    df::Union{DataFrame,Dict{String,DataFrame}};
    z_var::Union{Symbol,Nothing} = nothing,
    display_plot::Bool = false,
)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
        :payoff_mean_mean,
    ]

    if df isa DataFrame
        return plot_sweep_heatmap_Plots(
            df,
            :relatedness,
            :ext_pun,
            dependent_vars,
            z_var = z_var,
            display_plot = display_plot,
        )
    elseif df isa Dict{String,DataFrame}
        return plot_multiple_sweep_heatmap_Plots(
            df,
            :relatedness,
            :ext_pun0,
            dependent_vars,
            z_var = z_var,
        )
    end
end

function plot_sweep_rip_Plots(
    df::Union{DataFrame,Dict{String,DataFrame}};
    z_var::Union{Symbol,Nothing} = nothing,
    display_plot::Bool = false,
)
    dependent_vars =
        [:action_mean_mean, :norm_mean_mean, :ext_pun_mean_mean, :payoff_mean_mean]

    if df isa DataFrame
        return plot_sweep_heatmap_Plots(
            df,
            :relatedness,
            :int_pun_ext0,
            dependent_vars,
            z_var = z_var,
            display_plot = display_plot,
        )
    elseif df isa Dict{String,DataFrame}
        return plot_multiple_sweep_heatmap_Plots(
            df,
            :relatedness,
            :int_pun_ext0,
            dependent_vars,
            z_var = z_var,
        )
    end
end

function plot_sweep_rgs_Plots(
    df::Union{DataFrame,Dict{String,DataFrame}};
    display_plot::Bool = false,
)
    dependent_vars = [
        :action_mean_mean,
        :norm_mean_mean,
        :ext_pun_mean_mean,
        :int_pun_ext_mean_mean,
        :int_pun_self_mean_mean,
        :payoff_mean_mean,
    ]

    if df isa DataFrame
        return plot_sweep_heatmap_Plots(
            df,
            :relatedness,
            :group_size,
            dependent_vars,
            z_var = nothing,
            display_plot = display_plot,
        )
    elseif df isa Dict{String,DataFrame}
        return plot_multiple_sweep_heatmap_Plots(
            df,
            :relatedness,
            :group_size,
            dependent_vars,
            z_var = nothing,
        )
    end
end


################
# Compare Plots #################################################################################################################
################

function extract_plot_lists(plots_dict::Dict{String,T}) where {T<:Any}
    first_value = first(values(plots_dict))  # Check structure of first dictionary entry

    if first_value isa Vector{Plots.Plot}
        # Collect all Vector{Plots.Plot} entries into a single Vector{Vector{Plots.Plot}}
        return collect(values(plots_dict))
    elseif first_value isa Vector{Vector{Plots.Plot}}
        # Directly return the dictionary values as Vector{Vector{Plots.Plot}}, avoiding extra nesting
        return collect(values(plots_dict)...)
    else
        throw(ArgumentError("Unexpected data structure in plots_dict"))
    end
end

function compare_plot_lists(
    plot_lists::Vector{Vector{Plots.Plot}};
    xlim::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    ylim::Union{Nothing,Tuple{Float64,Float64}} = nothing,
)
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

        # Apply axis limits if specified
        if xlim !== nothing
            xlims!(p, xlim)
        end
        if ylim !== nothing
            ylims!(p, ylim)
        end

        display(p)
    end
end


###########
# PlotlyJS ######################################################################################################################
###########

function plot_simulation_Plotly(
    df::DataFrame,
    x_var::Symbol;
    dataset_name::String = "Simulation Data",
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
            "$(x_axis_x_varvariable)" .* ": " .* string.(df[!, x_var]) .* "<br>" .* trait .*
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
                x = df[!, x_var],
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
                x = df[!, x_var],
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
                x = df[!, x_var],
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
        title = dataset_name,
        xaxis_title = "$(x_var)",
        yaxis_title = "Traits",
        width = 600,
        height = 400,
        legend = Dict(:orientation => "h", :x => 0, :y => -0.2),
        hovermode = "x unified",
    )

    # Display plot
    display(p)
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
    dependent_vars =
        [:action_mean_mean, :norm_mean_mean, :ext_pun_mean_mean, :payoff_mean_mean]
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
