using Plots, PlotlyJS


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Plot Simulation Function
##################

# Function to set the GPU device for each worker
function set_device!(device_id)
    CUDA.device!(device_id)
    println("Assigned to GPU $device_id")
end

function run_simulation_on_gpu(parameters::SimulationParameters, replicate_id::Int64, device_id::Int)
    # Set the GPU for this worker
    set_device!(device_id)

    println("Running simulation replicate $replicate_id on GPU $device_id")

    # Run the simulation
    my_population = population_construction(parameters)
    my_simulation = simulation(my_population)

    # Group by generation and compute mean for each generation
    my_simulation_gdf = groupby(my_simulation, :generation)
    my_simulation_mean = combine(my_simulation_gdf,
                                 :action => mean,
                                 :a => mean,
                                 :p => mean,
                                 :T_ext => mean,
                                 :T_self => mean,
                                 :payoff => mean)

    # Add a column for replicate identifier
    rows_to_insert = nrow(my_simulation_mean)
    insertcols!(my_simulation_mean, 1, :replicate => fill(replicate_id, rows_to_insert))

    return my_simulation_mean
end

function simulation_replicate(parameters::SimulationParameters, num_replicates::Int64)
    # Assign GPU IDs to each replicate based on available GPUs
    gpu_ids = collect(0:length(CUDA.devices()) - 1)

    # Use pmap to parallelize the simulation across available GPUs and collect the results
    results = pmap(1:num_replicates) do i
        # Determine which GPU to assign (cycle through available GPUs)
        gpu_id = gpu_ids[(i - 1) % length(gpu_ids) + 1]
        # Run the simulation on the specific GPU and return the result
        run_simulation_on_gpu(parameters, i, gpu_id)
    end

    # Concatenate all the simulation means returned by each worker
    all_simulation_means = vcat(results...)

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
                    :T_ext_mean => mean => :T_ext_mean_mean,
                    :T_ext_mean => var => :T_ext_mean_var,
                    :T_self_mean => mean => :T_self_mean_mean,
                    :T_self_mean => var => :T_self_mean_var,
                    :payoff_mean => mean => :payoff_mean_mean,
                    :payoff_mean => var => :payoff_mean_var)

    # Compute standard deviation
    stats[:, "action_mean_std"] = sqrt.(stats[:, "action_mean_var"])
    stats[:, "a_mean_std"] = sqrt.(stats[:, "a_mean_var"])
    stats[:, "p_mean_std"] = sqrt.(stats[:, "p_mean_var"])
    stats[:, "T_ext_mean_std"] = sqrt.(stats[:, "T_ext_mean_var"])
    stats[:, "T_self_mean_std"] = sqrt.(stats[:, "T_self_mean_var"])
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
    p = Plots.plot()

    # Plot each replicate's data with consistent colors and labels
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])
        Plots.plot!(p, sim_data.generation, sim_data.action_mean, label="", color=colors["action"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.a_mean, label="", color=colors["a"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.p_mean, label="", color=colors["p"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.T_ext_mean, label="", color=colors["T_ext"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.T_self_mean, label="", color=colors["T_self"], linewidth=1, alpha=0.6)
        Plots.plot!(p, sim_data.generation, sim_data.payoff_mean, label="", color=colors["payoff"], linewidth=1, alpha=0.6)
    end

    # Plot mean and ribbons for each trait
    Plots.plot!(p, statistics.generation, statistics.action_mean_mean, ribbon=(statistics.action_mean_std, statistics.action_mean_std), label="action mean", color=colors["action mean"])
    Plots.plot!(p, statistics.generation, statistics.a_mean_mean, ribbon=(statistics.a_mean_std, statistics.a_mean_std), label="a mean", color=colors["a mean"])
    Plots.plot!(p, statistics.generation, statistics.p_mean_mean, ribbon=(statistics.p_mean_std, statistics.p_mean_std), label="p mean", color=colors["p mean"])
    Plots.plot!(p, statistics.generation, statistics.T_ext_mean_mean, ribbon=(statistics.T_ext_mean_std, statistics.T_ext_mean_std), label="T_ext mean", color=colors["T_ext mean"])
    Plots.plot!(p, statistics.generation, statistics.T_self_mean_mean, ribbon=(statistics.T_self_mean_std, statistics.T_self_mean_std), label="T_self mean", color=colors["T_self mean"])
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

    # Plot individual replicates
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])

        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.action_mean, mode="lines", line_color=colors["action"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.a_mean, mode="lines", line_color=colors["a"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.p_mean, mode="lines", line_color=colors["p"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.T_ext_mean, mode="lines", line_color=colors["T_ext"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.T_self_mean, mode="lines", line_color=colors["T_self"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
        add_trace!(p_replicates, PlotlyJS.scatter(x=sim_data.generation, y=sim_data.payoff_mean, mode="lines", line_color=colors["payoff"],
                                                 name="", opacity=0.6, showlegend=false, hoverinfo="none"))
    end

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
    for trait in ["action_mean", "a_mean", "p_mean", "T_ext_mean", "T_self_mean", "payoff_mean"]
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