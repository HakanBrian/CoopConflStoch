using BenchmarkTools, Plots


##################
# Game Functions
##################

include("CoopConflGameFuncs.jl")


##################
# Simulation Function
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

function plot_simulation_data(all_simulation_means::Tuple{DataFrame, Int64})
    num_replicates = all_simulation_means[2]

    # Define color palette for each trait type
    colors = Dict(
        "action" => :blue,
        "a" => :red,
        "p" => :green,
        "T" => :purple,
        "payoff" => :orange
    )

    # Initialize the plot
    p = plot(legend=false)

    # Plot each replicate's data with consistent colors and labels
    for i in 1:num_replicates
        sim_data = filter(row -> row.replicate == i, all_simulation_means[1])
        plot!(p, sim_data.generation, sim_data.action_mean, color=colors["action"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.a_mean, color=colors["a"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.p_mean, color=colors["p"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.T_mean, color=colors["T"], linewidth=1, alpha=0.6)
        plot!(p, sim_data.generation, sim_data.payoff_mean, color=colors["payoff"], linewidth=1, alpha=0.6)
    end

    # Customize and display the plot
    xlabel!("Generation")
    ylabel!("Traits")
    display(p)
end