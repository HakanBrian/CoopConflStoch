using BenchmarkTools, Plots


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Plot Simulation Function
##################

function simulation_replicate(my_parameter::simulation_parameters, num_replicates::Int64)
    # Container to hold mean data of each simulation
    output_length = floor(Int64, my_parameter.gmax/my_parameter.output_save_tick) * num_replicates
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
        my_population = population_construction(my_parameter)
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

function plot_simulation_data(all_simulation_means::Tuple{DataFrame, Int64})
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
    p = plot()

    # Plot each replicate's data with consistent colors and labels
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])
        plot!(p, sim_data.generation, sim_data.action_mean, label="", color=colors["action"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.a_mean, label="", color=colors["a"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.p_mean, label="", color=colors["p"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.T_mean, label="", color=colors["T"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.payoff_mean, label="", color=colors["payoff"], linewidth=1, alpha=0.6)
    end

    # Plot mean and ribbons for each trait
    plot!(p, statistics.generation, statistics.action_mean_mean, ribbon=(statistics.action_mean_std, statistics.action_mean_std), label="action mean", color=colors["action mean"])
    plot!(p, statistics.generation, statistics.a_mean_mean, ribbon=(statistics.a_mean_std, statistics.a_mean_std), label="a mean", color=colors["a mean"])
    plot!(p, statistics.generation, statistics.p_mean_mean, ribbon=(statistics.p_mean_std, statistics.p_mean_std), label="p mean", color=colors["p mean"])
    plot!(p, statistics.generation, statistics.T_mean_mean, ribbon=(statistics.T_mean_std, statistics.T_mean_std), label="T mean", color=colors["T mean"])
    plot!(p, statistics.generation, statistics.payoff_mean_mean, ribbon=(statistics.payoff_mean_std, statistics.payoff_mean_std), label="payoff mean", color=colors["payoff mean"])

    # Display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display(p)
end