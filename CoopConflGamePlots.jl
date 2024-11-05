using Distributed, Plots, PlotlyJS, CSV, FilePathsBase


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Plot Simulation Function
##################

function run_simulation(parameters::SimulationParameters, param_id::Int64, replicate_id::Int64)
    println("Running simulation replicate $replicate_id for param_id $param_id")

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

    # Add columns for replicate and param_id
    rows_to_insert = nrow(my_simulation_mean)
    insertcols!(my_simulation_mean, 1, :param_id => fill(param_id, rows_to_insert))
    insertcols!(my_simulation_mean, 2, :replicate => fill(replicate_id, rows_to_insert))

    return my_simulation_mean
end

function simulation_replicate(parameters::SimulationParameters, num_replicates::Int64)
    # Use pmap to parallelize the simulation
    results = pmap(1:num_replicates) do i
        run_simulation(parameters, 1, i)
    end

    # Concatenate all the simulation means returned by each worker
    all_simulation_means = vcat(results...)

    return all_simulation_means
end

function simulation_replicate(parameter_sweep::Vector{SimulationParameters}, num_replicates::Int64)
    # Create a list of tasks (parameter set index, parameter set, replicate) to distribute
    tasks = [(idx, parameters, replicate) for (idx, parameters) in enumerate(parameter_sweep) for replicate in 1:num_replicates]

    # Use pmap to distribute the tasks across the workers
    results = pmap(tasks) do task
        param_idx, parameters, replicate = task
        # Run simulation and store the result with the parameter set index
        run_simulation(parameters, param_idx, replicate)
    end

    # Concatenate all results into a single DataFrame
    all_simulation_means = vcat(results...)

    return all_simulation_means
end

function calculate_statistics(all_simulation_means::DataFrame)
    # Group by generation
    grouped = groupby(all_simulation_means, :generation)

    # Calculate mean and standard deviation for each trait across replicates
    stats = combine(grouped,
                    :action_mean => mean => :action_mean_mean,
                    :action_mean => std => :action_mean_std,
                    :a_mean => mean => :a_mean_mean,
                    :a_mean => std => :a_mean_std,
                    :p_mean => mean => :p_mean_mean,
                    :p_mean => std => :p_mean_std,
                    :T_ext_mean => mean => :T_ext_mean_mean,
                    :T_ext_mean => std => :T_ext_mean_std,
                    :T_self_mean => mean => :T_self_mean_mean,
                    :T_self_mean => std => :T_self_mean_std,
                    :payoff_mean => mean => :payoff_mean_mean,
                    :payoff_mean => std => :payoff_mean_std)

    return stats
end

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
        Plots.plot!(p, statistics.generation, statistics.action_mean_mean, ribbon=(statistics.action_mean_std, statistics.action_mean_std), 
                    label="action ($i)", color=colors["action mean"])
        Plots.plot!(p, statistics.generation, statistics.a_mean_mean, ribbon=(statistics.a_mean_std, statistics.a_mean_std), 
                    label="a ($i)", color=colors["a mean"])
        Plots.plot!(p, statistics.generation, statistics.p_mean_mean, ribbon=(statistics.p_mean_std, statistics.p_mean_std), 
                    label="p ($i)", color=colors["p mean"])
        Plots.plot!(p, statistics.generation, statistics.T_ext_mean_mean, ribbon=(statistics.T_ext_mean_std, statistics.T_ext_mean_std), 
                    label="T_ext ($i)", color=colors["T_ext mean"])
        Plots.plot!(p, statistics.generation, statistics.T_self_mean_mean, ribbon=(statistics.T_self_mean_std, statistics.T_self_mean_std), 
                    label="T_self ($i)", color=colors["T_self mean"])
        Plots.plot!(p, statistics.generation, statistics.payoff_mean_mean, ribbon=(statistics.payoff_mean_std, statistics.payoff_mean_std), 
                    label="payoff ($i)", color=colors["payoff mean"])
    end

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display("image/png", p)
end

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

function save_simulation(simulation::DataFrame, filepath::String)
    # Ensure the filepath has the .csv extension
    if !endswith(filepath, ".csv")
        filepath *= ".csv"
    end

    # Convert to an absolute path (in case it's not already)
    filepath = abspath(filepath)

    # Check if the file already exists, and print a warning if it does
    if isfile(filepath)
        println("Warning: File '$filepath' already exists and will be overwritten.")
    end

    # Save the dataframe, overwriting the file if it exists
    CSV.write(filepath, simulation)
    println("File saved as: $filepath")
end

function read_simulation(filepath::String)
    # Ensure the filepath has the .csv extension
    if !endswith(filepath, ".csv")
        filepath *= ".csv"
    end

    # Convert to an absolute path (in case it's not already)
    filepath = abspath(filepath)

    # Check if the file exists before attempting to read it
    if !isfile(filepath)
        error("File '$filepath' does not exist.")
    else
        # Read the CSV file into a DataFrame
        simulation = CSV.read(filepath, DataFrame)
        println("File successfully loaded from: $filepath")
        return simulation
    end
end