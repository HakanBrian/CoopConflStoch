using BenchmarkTools, Plots, PlotlyJS


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Plot Simulation Function
##################

function simulation_replicate(parameters::SimulationParameters, num_replicates::Int64)
    # Container to hold mean data of each simulation
    output_length = floor(Int64, parameters.gmax/parameters.output_save_tick) * num_replicates
    all_simulation_means = DataFrame(
        replicate = Vector{Int64}(undef, output_length),
        generation = Vector{Int64}(undef, output_length),
        action_mean = Vector{Float64}(undef, output_length),
        a_mean = Vector{Float64}(undef, output_length),
        p_mean = Vector{Float64}(undef, output_length),
        T_mean = Vector{Float64}(undef, output_length),
        payoff_mean = Vector{Float64}(undef, output_length)
    )

    # Index to keep track of where to insert rows
    row_index = 1

    for i in 1:num_replicates
        println("Running simulation replicate $i")

        # Run the simulation
        my_population = population_construction(parameters)
        my_simulation = simulation(my_population)

        # Group by generation and compute mean for each generation
        my_simulation_gdf = groupby(my_simulation, :generation)
        my_simulation_mean = combine(my_simulation_gdf,
                                    :action => mean,
                                    :a => mean,
                                    :p => mean,
                                    :T => mean,
                                    :payoff => mean)

        # Add a column for replicate identifier
        rows_to_insert = nrow(my_simulation_mean)
        insertcols!(my_simulation_mean, 1, :replicate => fill(i, rows_to_insert))

        # Insert rows into preallocated DataFrame
        all_simulation_means[row_index:row_index + rows_to_insert - 1, :] .= my_simulation_mean
        row_index += rows_to_insert
    end

    return all_simulation_means, num_replicates
end

function calculate_statistics(all_simulation_means::DataFrame)
    # Group by generation
    grouped = groupby(all_simulation_means, :generation)

    # Calculate mean and variance for each trait across replicates
    stats = combine(grouped,
                    :action_mean => mean => :action_mean_mean,
                    :action_mean => var => :action_mean_var,
                    :a_mean => mean => :a_mean_mean,
                    :a_mean => var => :a_mean_var,
                    :p_mean => mean => :p_mean_mean,
                    :p_mean => var => :p_mean_var,
                    :T_mean => mean => :T_mean_mean,
                    :T_mean => var => :T_mean_var,
                    :payoff_mean => mean => :payoff_mean_mean,
                    :payoff_mean => var => :payoff_mean_var)

    # Compute standard deviation
    stats[:, "action_mean_std"] = sqrt.(stats[:, "action_mean_var"])
    stats[:, "a_mean_std"] = sqrt.(stats[:, "a_mean_var"])
    stats[:, "p_mean_std"] = sqrt.(stats[:, "p_mean_var"])
    stats[:, "T_mean_std"] = sqrt.(stats[:, "T_mean_var"])
    stats[:, "payoff_mean_std"] = sqrt.(stats[:, "payoff_mean_var"])

    return stats
end

function plot_simulation_data_Plots(all_simulation_means::Tuple{DataFrame, Int64})
    num_replicates = all_simulation_means[2]

    # Calculate statistics
    statistics = calculate_statistics(all_simulation_means[1])

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T" => :purple,
        "payoff" => :orange,
        "action mean" => :blue4,
        "a mean" => :red4,
        "p mean" => :green4,
        "T mean" => :purple4,
        "payoff mean" => :orange4
    )

    # Initialize the plot
    p = Plots.plot()

    # Plot each replicate's data with consistent colors and labels
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])
        Plots.plot!(p, sim_data.generation, sim_data.action_mean, label="", color=colors["action"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.a_mean, label="", color=colors["a"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.p_mean, label="", color=colors["p"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.T_mean, label="", color=colors["T"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.payoff_mean, label="", color=colors["payoff"], linewidth=1, alpha=0.6)
    end

    # Plot mean and ribbons for each trait
    Plots.plot!(p, statistics.generation, statistics.action_mean_mean, ribbon=(statistics.action_mean_std, statistics.action_mean_std), label="action mean", color=colors["action mean"])
    Plots.plot!(p, statistics.generation, statistics.a_mean_mean, ribbon=(statistics.a_mean_std, statistics.a_mean_std), label="a mean", color=colors["a mean"])
    Plots.plot!(p, statistics.generation, statistics.p_mean_mean, ribbon=(statistics.p_mean_std, statistics.p_mean_std), label="p mean", color=colors["p mean"])
    Plots.plot!(p, statistics.generation, statistics.T_mean_mean, ribbon=(statistics.T_mean_std, statistics.T_mean_std), label="T mean", color=colors["T mean"])
    Plots.plot!(p, statistics.generation, statistics.payoff_mean_mean, ribbon=(statistics.payoff_mean_std, statistics.payoff_mean_std), label="payoff mean", color=colors["payoff mean"])

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display(p)
end

function plot_simulation_data_Plotly(all_simulation_means::Tuple{DataFrame, Int64})
    num_replicates = all_simulation_means[2]
    statistics = calculate_statistics(all_simulation_means[1])

    p_means = Plot()
    p_replicates = Plot()

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T" => :purple,
        "payoff" => :orange,
        "action_stdev" => "rgba(0,0,255,0.2)",
        "a_stdev" => "rgba(255,0,0,0.2)",
        "p_stdev" => "rgba(0,255,0,0.2)",
        "T_stdev" => "rgba(128,0,128,0.2)",
        "payoff_stdev" => "rgba(255,165,0,0.2)",
    )

    # Plot individual replicates
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])

        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.action_mean, mode="lines", line_color=colors["action"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.a_mean, mode="lines", line_color=colors["a"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.p_mean, mode="lines", line_color=colors["p"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.T_mean, mode="lines", line_color=colors["T"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.payoff_mean, mode="lines", line_color=colors["payoff"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
    end

    # Create formatted hover text for each trait
    statistics[!, :action_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>action Mean: " .* string.(statistics.action_mean_mean) .* "<br>Std Dev: " .* string.(statistics.action_mean_std)
    statistics[!, :a_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>a Mean: " .* string.(statistics.a_mean_mean) .* "<br>Std Dev: " .* string.(statistics.a_mean_std)
    statistics[!, :p_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>p Mean: " .* string.(statistics.p_mean_mean) .* "<br>Std Dev: " .* string.(statistics.p_mean_std)
    statistics[!, :T_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>T Mean: " .* string.(statistics.T_mean_mean) .* "<br>Std Dev: " .* string.(statistics.T_mean_std)
    statistics[!, :payoff_mean_hover] = "Generation: " .* string.(statistics.generation) .* "<br>payoff Mean: " .* string.(statistics.payoff_mean_mean) .* "<br>Std Dev: " .* string.(statistics.payoff_mean_std)

    # Plot replicate means with ribbons for standard deviation
    for trait in ["action", "a", "p", "T", "payoff"]
        add_trace!(p_means, PlotlyJS.scatter(x=statistics.generation, y=statistics[!, trait * "_mean_mean"],
                                            mode="lines", line_color=colors[trait], name=trait * " Mean",
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

    # Layout for individual replicates
    relayout!(p_replicates, title="Individual Replicates",
            xaxis_title="Generation", yaxis_title="Traits",
            legend=Dict(:orientation => "h", :x => 0, :y => -0.2), hovermode="x unified")

    # Layout for replicate means
    relayout!(p_means, title="Mean of Replicates",
            xaxis_title="Generation", yaxis_title="Traits",
            legend=Dict(:orientation => "h", :x => 0, :y => -0.2), hovermode="x unified")

    # Display plots
    display(p_replicates)
    display(p_means)

    # Prepare table data
    table_data = []
    for trait in ["action_mean", "a_mean", "p_mean", "T_mean", "payoff_mean"]
        start_values = statistics[1, [trait * "_mean"]][1]
        end_values = statistics[end, [trait * "_mean"]][1]
        push!(table_data, (trait, start_values, end_values))
    end

    # Create table
    table_df = DataFrame(table_data, [:Trait, :Start_Value, :End_Value])
    table_trace = PlotlyJS.table(
        header = Dict(:values => ["Trait", "Start Value", "End Value"]),
        cells = Dict(:values => [table_df.Trait, table_df.Start_Value, table_df.End_Value])
    )

    table_layout = Layout(
        title="Beginning and Final Values",
        margin=Dict(:t => 50, :b => 50),
        height=300
    )

    # Display table
    display(Plot([table_trace], table_layout))
end