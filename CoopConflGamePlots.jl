using Interpolations, Plots, PlotlyJS


##################
# Helper Functions
##################

include("CoopConflGameHelper.jl")


##################
# Plot Function
##################

function plot_simulation_data_Plots(all_simulation_means::DataFrame; param_id::Union{Nothing, Int64}=nothing)
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
        "payoff mean" => :orange4
    )

    # Initialize the plot
    p = Plots.plot(legend=false)

    for i in 1:num_params
        # Filter by param_id
        param_data = filter(row -> row.param_id == i, all_simulation_means)

        # Calculate statistics for the current parameter set
        statistics = calculate_statistics(param_data)

        # Plot mean and ribbons for each trait with a distinct label for each parameter set
        for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
            Plots.plot!(p,
                        statistics.generation,
                        statistics[!, trait * "_mean_mean"],
                        ribbon=(statistics[!, trait * "_mean_std"], statistics[!, trait * "_mean_std"]), 
                        label=trait * " ($i)",
                        color=colors[trait * " mean"])
        end
    end

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display("image/png", p)
end

function plot_full_sweep_Plots(statistics::DataFrame)
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
        "payoff mean" => :orange4
    )

    plot_var_set = [["action", "a", "p", "T_ext", "T_self", "payoff"], ["p", "T_ext", "T_self"], ["p"], ["action", "a"]]

    for plot_var in plot_var_set
        # Initialize plot
        p = Plots.plot(legend=true)

        # Plot mean and ribbons for each trait
        for trait in plot_var
            Plots.plot!(p,
                        statistics.relatedness,
                        statistics[!, trait * "_mean_mean"],
                        ribbon=(statistics[!, trait * "_mean_std"], statistics[!, trait * "_mean_std"]), 
                        label=trait,
                        color=colors[trait * " mean"])
        end

        # Display the plot
        xlabel!("Relatedness")
        ylabel!("Value")
        display("image/png", p)
    end
end

function plot_sweep_rep_Plots(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ep_values = sort(unique(statistics.ext_pun))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :ext_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `ext_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:ext_pun)])

        # Plot heatmap
        p = Plots.heatmap(r_values, ep_values, heatmap_matrix,
                            color=:viridis,
                            xlabel="Relatedness",
                            ylabel="External Punishment",
                            title="Heatmap of $var",
                            colorbar_title="Value")

        display("image/png", p)
    end
end

function plot_sweep_rip_Plots(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ip_values = sort(unique(statistics.int_pun))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :int_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `int_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:int_pun)])

        # Plot heatmap
        p = Plots.heatmap(r_values, ip_values, heatmap_matrix,
                            color=:viridis,
                            xlabel="Relatedness",
                            ylabel="Internal Punishment",
                            title="Heatmap of $var",
                            colorbar_title="Value")

        display("image/png", p)
    end
end

function plot_sweep_rgs_Plots(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    gs_values = sort(unique(statistics.group_size))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :group_size, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `group_size` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:group_size)])

        # Plot heatmap
        p = Plots.heatmap(r_values, gs_values, heatmap_matrix,
                            color=:viridis,
                            xlabel="Relatedness",
                            ylabel="Group Size",
                            title="Heatmap of $var",
                            colorbar_title="Value")

        display("image/png", p)
    end
end

function plot_sweep_rep_smooth_Plots(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ep_values = sort(unique(statistics.ext_pun))

    # Define grid fineness
    r_length = 10*length(r_values)
    ep_length = 10*length(ep_values)

    # Define a finer grid for smoothing
    r_fine = range(minimum(r_values), maximum(r_values), length=r_length)  # Fine relatedness grid
    ep_fine = range(minimum(ep_values), maximum(ep_values), length=ep_length)  # Fine external punishment grid

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :ext_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `ext_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:ext_pun)])

        # Create the interpolator using grid dimensions and matrix
        interp = interpolate((1:size(heatmap_matrix, 2), 1:size(heatmap_matrix, 1)), heatmap_matrix, Gridded(Linear()))

        # Evaluate interpolation on the finer grid
        interp_r_fine = range(1, size(heatmap_matrix, 2), length=r_length)
        interp_ep_fine = range(1, size(heatmap_matrix, 1), length=ep_length)

        # Smooth the data based on interpolator
        heatmap_smooth = [interp(i, j) for i in interp_r_fine, j in interp_ep_fine]

        # Plot heatmap
        p = Plots.heatmap(r_fine, ep_fine, heatmap_smooth, color=:viridis, xlabel="Relatedness", ylabel="External Punishment",
                          title="Heatmap of $var", colorbar_title="Value")

        # Add contour lines
        contour!(r_fine, ep_fine, heatmap_smooth, levels=10, color=:white, linewidth=0.8)

        display("image/png", p)
    end
end


##################
# Plotly Function
##################

function create_trait_table(all_simulation_means::DataFrame)
    table_data = []

    # Group by param_id and extract start and end values for each group
    grouped_by_param = groupby(all_simulation_means, :param_id)

    for group in grouped_by_param
        param_id_val = group.param_id[1]  # Extract the param_id value
        for trait in ["action_mean", "a_mean", "p_mean", "T_ext_mean", "T_self_mean", "payoff_mean"]
            start_values = first(group)[trait]  # Start value (first generation)
            end_values = last(group)[trait]  # End value (last generation)
            push!(table_data, (param_id_val, trait, start_values, end_values))
        end
    end

    # Create a DataFrame for the table with Param_ID
    table_df = DataFrame(table_data, [:Param_ID, :Trait, :Start_Value, :End_Value])

    # Create table trace
    table_trace = PlotlyJS.table(
        header = Dict(:values => ["Param ID", "Trait", "Start Value", "End Value"]),
        cells = Dict(:values => [table_df.Param_ID, table_df.Trait, table_df.Start_Value, table_df.End_Value])
    )

    # Set table layout
    table_layout = Layout(
        title="Beginning and Final Values",
        margin=Dict(:t => 50, :b => 50),
        height=300
    )

    # Display the table
    display(Plot([table_trace], table_layout))
end

function plot_simulation_data_Plotly(all_simulation_means::DataFrame; param_id::Union{Nothing, Int64}=nothing)
    # Filter the data if param_id is provided
    if param_id !== nothing
        all_simulation_means = filter(row -> row.param_id == param_id, all_simulation_means)
    end

    # Determine the number of params and replicates
    num_params = maximum(all_simulation_means.param_id)  # Assuming param numbers are consistent

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
        statistics[!, :action_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>action Mean: " .* string.(statistics.action_mean_mean) .* "<br>Std Dev: " .* string.(statistics.action_mean_std)
        statistics[!, :a_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>a Mean: " .* string.(statistics.a_mean_mean) .* "<br>Std Dev: " .* string.(statistics.a_mean_std)
        statistics[!, :p_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>p Mean: " .* string.(statistics.p_mean_mean) .* "<br>Std Dev: " .* string.(statistics.p_mean_std)
        statistics[!, :T_ext_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>T_ext Mean: " .* string.(statistics.T_ext_mean_mean) .* "<br>Std Dev: " .* string.(statistics.T_ext_mean_std)
        statistics[!, :T_self_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>T_self Mean: " .* string.(statistics.T_self_mean_mean) .* "<br>Std Dev: " .* string.(statistics.T_self_mean_std)
        statistics[!, :payoff_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>payoff Mean: " .* string.(statistics.payoff_mean_mean) .* "<br>Std Dev: " .* string.(statistics.payoff_mean_std)

        # Plot replicate means with ribbons for standard deviation
        for trait in ["action", "a", "p", "T_ext", "T_self", "payoff"]
            add_trace!(p_means, PlotlyJS.scatter(x=statistics.generation, y=statistics[!, trait * "_mean_mean"],
                                                mode="lines", line_color=colors[trait], name=trait * " ($i)",
                                                hovertext=statistics[!, trait * "_mean_hover"],
                                                hoverinfo="text"))

            add_trace!(p_means, PlotlyJS.scatter(x=statistics.generation, y=statistics[!, trait * "_mean_mean"] .+ statistics[!, trait * "_mean_std"],
                                                mode="lines", line_color=colors[trait], name="", fill="tonexty",
                                                fillcolor=colors[trait * "_stdev"], line=Dict(:width => 0),
                                                hoverinfo="none", showlegend=false))

            add_trace!(p_means, PlotlyJS.scatter(x=statistics.generation, y=statistics[!, trait * "_mean_mean"] .- statistics[!, trait * "_mean_std"],
                                                mode="lines", line_color=colors[trait], name="", fill="tonexty",
                                                fillcolor=colors[trait * "_stdev"], line=Dict(:width => 0),
                                                hoverinfo="none", showlegend=false))
        end
    end

    # Layout for replicate means
    relayout!(p_means, title="Mean of Replicates",
            xaxis_title="Generation", yaxis_title="Traits",
            legend=Dict(:orientation => "h", :x => 0, :y => -0.2), hovermode="x unified")

    # Display plots
    display(p_means)
end

function plot_full_sweep_Plotly(statistics::DataFrame)
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
    statistics[!, :action_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>action Mean: " .* string.(statistics.action_mean_mean) .* "<br>Std Dev: " .* string.(statistics.action_mean_std)
    statistics[!, :a_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>a Mean: " .* string.(statistics.a_mean_mean) .* "<br>Std Dev: " .* string.(statistics.a_mean_std)
    statistics[!, :p_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>p Mean: " .* string.(statistics.p_mean_mean) .* "<br>Std Dev: " .* string.(statistics.p_mean_std)
    statistics[!, :T_ext_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>T_ext Mean: " .* string.(statistics.T_ext_mean_mean) .* "<br>Std Dev: " .* string.(statistics.T_ext_mean_std)
    statistics[!, :T_self_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>T_self Mean: " .* string.(statistics.T_self_mean_mean) .* "<br>Std Dev: " .* string.(statistics.T_self_mean_std)
    statistics[!, :payoff_mean_hover] = "Relatedness: " .* string.(statistics.relatedness) .* "<br>payoff Mean: " .* string.(statistics.payoff_mean_mean) .* "<br>Std Dev: " .* string.(statistics.payoff_mean_std)

    plot_var_set = [["action", "a", "p", "T_ext", "T_self", "payoff"], ["p", "T_ext", "T_self"], ["p"], ["action", "a"]]

    for plot_var in plot_var_set
        # Initialize plot
        p = Plot()

        # Plot replicate means with ribbons for standard deviation
        for trait in plot_var
            add_trace!(p, PlotlyJS.scatter(x=statistics.relatedness, y=statistics[!, trait * "_mean_mean"],
                                            mode="lines", line_color=colors[trait], name=trait,
                                            hovertext=statistics[!, trait * "_mean_hover"],
                                            hoverinfo="text"))

            add_trace!(p, PlotlyJS.scatter(x=statistics.relatedness, y=statistics[!, trait * "_mean_mean"] .+ statistics[!, trait * "_mean_std"],
                                            mode="lines", line_color=colors[trait], name="", fill="tonexty",
                                            fillcolor=colors[trait * "_stdev"], line=Dict(:width => 0),
                                            hoverinfo="none", showlegend=false))

            add_trace!(p, PlotlyJS.scatter(x=statistics.relatedness, y=statistics[!, trait * "_mean_mean"] .- statistics[!, trait * "_mean_std"],
                                            mode="lines", line_color=colors[trait], name="", fill="tonexty",
                                            fillcolor=colors[trait * "_stdev"], line=Dict(:width => 0),
                                            hoverinfo="none", showlegend=false))
        end

        # Layout for replicate means
        relayout!(p, title="Mean of Replicates",
                xaxis_title="Relatedness", yaxis_title="Traits",
                legend=Dict(:orientation => "h", :x => 0, :y => -0.2), hovermode="x unified")

        # Display plots
        display(p)
    end
end

function plot_sweep_rep_Plotly(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ep_values = sort(unique(statistics.ext_pun))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :ext_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `ext_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:ext_pun)])

        # Create a heatmap trace
        trace = PlotlyJS.heatmap(
            z=heatmap_matrix,  # Data matrix
            x=r_values,  # Relatedness (x-axis)
            y=ep_values,  # External Punishment (y-axis)
            colorscale="Viridis",
            colorbar_title="Value"
        )

        # Add layout
        layout = Layout(
            title="Heatmap of $var",
            xaxis_title="Relatedness",
            yaxis_title="External Punishment"
        )

        # Plot
        display(PlotlyJS.plot([trace], layout))
    end
end

function plot_sweep_rip_Plotly(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ip_values = sort(unique(statistics.int_pun))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :int_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `int_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:int_pun)])

        # Create a heatmap trace
        trace = PlotlyJS.heatmap(
            z=heatmap_matrix,  # Data matrix
            x=r_values,  # Relatedness (x-axis)
            y=ip_values,  # External Punishment (y-axis)
            colorscale="Viridis",
            colorbar_title="Value"
        )

        # Add layout
        layout = Layout(
            title="Heatmap of $var",
            xaxis_title="Relatedness",
            yaxis_title="Internal Punishment"
        )

        # Plot
        display(PlotlyJS.plot([trace], layout))
    end
end

function plot_sweep_rgs_Plotly(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :p_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    gs_values = sort(unique(statistics.group_size))

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :group_size, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `group_size` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:group_size)])

        # Create a heatmap trace
        trace = PlotlyJS.heatmap(
            z=heatmap_matrix,  # Data matrix
            x=r_values,  # Relatedness (x-axis)
            y=gs_values,  # External Punishment (y-axis)
            colorscale="Viridis",
            colorbar_title="Value"
        )

        # Add layout
        layout = Layout(
            title="Heatmap of $var",
            xaxis_title="Relatedness",
            yaxis_title="Group Size"
        )

        # Plot
        display(PlotlyJS.plot([trace], layout))
    end
end

function plot_sweep_rep_smooth_Plotly(statistics::DataFrame)
    # List of dependent variables to plot as separate heatmaps
    dependent_vars = [:action_mean_mean, :a_mean_mean, :T_ext_mean_mean, :T_self_mean_mean, :payoff_mean_mean]

    # Get r_values and ep_values dynamically
    r_values = sort(unique(statistics.relatedness))
    ep_values = sort(unique(statistics.ext_pun))

    # Define grid fineness
    r_length = 10 * length(r_values)
    ep_length = 10 * length(ep_values)

    # Define a finer grid for smoothing
    r_fine = range(minimum(r_values), maximum(r_values), length=r_length)  # Fine relatedness grid
    ep_fine = range(minimum(ep_values), maximum(ep_values), length=ep_length)  # Fine external punishment grid

    for var in dependent_vars
        # Pivot the data for the current dependent variable
        heatmap_data = unstack(statistics, :ext_pun, :relatedness, var)

        # Convert the DataFrame to a matrix (remove `ext_pun` column)
        heatmap_matrix = Matrix{Float64}(heatmap_data[!, Not(:ext_pun)])

        # Create the interpolator using grid dimensions and matrix
        interp = interpolate((1:size(heatmap_matrix, 2), 1:size(heatmap_matrix, 1)), heatmap_matrix, Gridded(Linear()))

        # Evaluate interpolation on the finer grid
        interp_r_fine = range(1, size(heatmap_matrix, 2), length=r_length)
        interp_ep_fine = range(1, size(heatmap_matrix, 1), length=ep_length)

        # Smooth the data based on interpolator
        heatmap_smooth = [interp(i, j) for i in interp_r_fine, j in interp_ep_fine]

        # Create heatmap trace
        heatmap_trace = PlotlyJS.heatmap(
            z=heatmap_smooth,  # Smoothed data matrix
            x=r_fine,  # Fine relatedness grid
            y=ep_fine,  # Fine external punishment grid
            colorscale="Viridis",
            colorbar_title="Value"
        )

        # Create contour trace
        contour_trace = PlotlyJS.contour(
            z=heatmap_smooth,  # Smoothed data matrix
            x=r_fine,  # Fine relatedness grid
            y=ep_fine,  # Fine external punishment grid
            colorscale="Viridis",
            contours_coloring="lines",
            line_width=0.8,
            line_color="white",
            ncontours=10  # Number of contour levels
        )

        # Add layout
        layout = Layout(
            title="Heatmap of $var with Contours",
            xaxis_title="Relatedness",
            yaxis_title="External Punishment"
        )

        # Combine traces and plot
        display(PlotlyJS.plot([heatmap_trace, contour_trace], layout))
    end
end