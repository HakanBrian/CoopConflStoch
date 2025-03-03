module IOHandler

export save_simulation,
    read_simulation,
    generate_filename_suffix,
    modify_filename,
    log_metadata,
    update_dataset_params!,
    process_simulation

using ..MainSimulation.SimulationParameters
import ..MainSimulation.SimulationParameters: SimulationParameter

using CSV, JSON3, Dates, DataFrames

function save_simulation(simulation::DataFrame, filepath::String)
    # Ensure the filepath has the .csv extension
    if !endswith(filepath, ".csv")
        filepath *= ".csv"
    end

    # Convert to an absolute path
    filepath = abspath(filepath)

    # Extract the directory path from the filename
    dir_path = dirname(filepath)

    # Check if the directory exists
    if !isdir(dir_path)
        error(
            "Error: Directory '$dir_path' does not exist. Please create it before saving.",
        )
    end

    # Check if the file already exists and warn the user
    if isfile(filepath)
        @warn "File '$filepath' already exists and will be overwritten."
    end

    # Save the DataFrame to a CSV file
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

function generate_filename_suffix(
    param_dict::Dict{Symbol,<:Number},
    condition::String = "Filtered";
    time_point::Union{Nothing,Int} = nothing,
)
    # Lexicographic sorting
    sorted_keys = sort(collect(keys(param_dict)))

    # Convert parameters to key-value format
    if condition == "Full"
        param_str = join(["$(k)=$(param_dict[k])" for k in sorted_keys], "_")
    elseif condition == "Filtered"
        param_str = join(["$(k)" for k in sorted_keys], "_")
    end

    # Add condition
    suffix = "$(param_str)_$(condition)"

    # Append time point if applicable
    if !isnothing(time_point)
        suffix *= "_G$(time_point)"
    end

    return suffix
end

function modify_filename(filepath::String, key::String)
    dir, filename = splitdir(filepath)
    base, ext = splitext(filename)

    # If no extension, assume ".csv"
    if ext == ""
        ext = ".csv"
    end

    # Construct the new filepath with suffix
    new_filepath = joinpath(dir, base * "_" * key * ext)

    # Ensure forward slashes for consistency
    return replace(new_filepath, "\\" => "/")
end

function log_metadata(
    filename::String,
    sweep_vars::Dict{Symbol,AbstractVector},
    parameters::SimulationParameter,  # Actual simulation parameters
    condition::String;
    save_generations::Union{Nothing, Vector{Real}}=nothing,
    metadata_file::String="$(filename)_metadata.json",
    notes::String=""  # Optional notes field
)
    # Function to format parameter values while preserving step size
    function format_param_value(values)
        if isa(values, AbstractVector) && length(values) > 1 && eltype(values) <: Real
            steps = diff(values)  # Compute step differences

            if length(unique(steps)) == 1  # Uniform step size
                step_size = steps[1]
                return "$(minimum(values)):$(maximum(values)):$(step_size)"
            else
                # Detect step segments
                segments = []
                start_idx = 1

                while start_idx <= length(values)
                    if start_idx == length(values)  # Single value case
                        push!(segments, string(values[start_idx]))
                        break
                    end

                    step = values[start_idx + 1] - values[start_idx]
                    segment = [values[start_idx]]

                    # Extend segment while step size remains constant
                    for i in start_idx+1:length(values)-1
                        if values[i+1] - values[i] == step
                            push!(segment, values[i])
                        else
                            break
                        end
                    end
                    push!(segment, values[start_idx + length(segment) - 1])

                    # Store segment as "min:max:step" if >2 elements
                    if length(segment) > 2
                        push!(segments, "$(segment[1]):$(segment[end]):$(step)")
                    else
                        push!(segments, join(segment, ", "))  # List values if short
                    end

                    start_idx += length(segment)  # Move to next section
                end

                return join(segments, ", ")  # Format mixed step description
            end
        else
            return string(values)  # Single value
        end
    end

    # Function to classify traits based on mutation status and initial value
    function classify_trait(initial_value, mutation_enabled)
        if mutation_enabled
            return "Mutable"
        elseif initial_value == 0
            return "Turned Off"
        else
            return "Fixed"
        end
    end

    # Store all sweep_vars in "parameters" while keeping step size format
    formatted_parameters = Dict(string(k) => format_param_value(v) for (k, v) in sweep_vars)

    # Track mutation-based treatment classification
    mutation_treatment = Dict()

    # Check mutation settings - using actual `parameters`
    for (trait, mutation_flag) in [
        (:norm0, :norm_mutation_enabled),
        (:ext_pun0, :ext_pun_mutation_enabled),
        (:int_pun_ext0, :int_pun_ext_mutation_enabled),
        (:int_pun_self0, :int_pun_self_mutation_enabled)
    ]
        initial_value = getfield(parameters, trait)
        mutation_status = getfield(parameters, mutation_flag)

        if haskey(sweep_vars, trait) || haskey(sweep_vars, mutation_flag)
            mutation_treatment[string(trait)] = classify_trait(initial_value, mutation_status)
        end
    end

    # Construct formatted treatment description
    treatment_desc = if isempty(mutation_treatment)
        "Default"
    else
        "{" * join(["$k: $(v)" for (k, v) in mutation_treatment], ", ") * "}"
    end

    # Convert time points
    time_str = isnothing(save_generations) ? "All" : save_generations

    # Create metadata entry
    entry = Dict(
        "filename" => filename,
        "parameters" => formatted_parameters,  # Keeps step size formatting
        "condition" => condition,
        "saved_time_points" => time_str,
        "treatment" => treatment_desc,  # Auto-generated treatment label
        "date" => string(Dates.today()),
        "notes" => notes  # Include optional notes field
    )

    # Load existing metadata if the file exists
    if isfile(metadata_file)
        metadata = JSON3.read(read(metadata_file, String))
    else
        metadata = Dict("datasets" => [])
    end

    # Append new metadata entry
    push!(metadata["datasets"], entry)

    # Save updated metadata
    open(metadata_file, "w") do f
        write(f, JSON3.write(metadata, indent=4))  # Pretty-print JSON
    end
end

function update_dataset_params!(df::DataFrame)
    rename!(df, :a_mean_mean => :norm_mean_mean)
    rename!(df, :a_mean_std => :norm_mean_std)
    rename!(df, :p_mean_mean => :ext_pun_mean_mean)
    rename!(df, :p_mean_std => :ext_pun_mean_std)
    rename!(df, :T_ext_mean_mean => :int_pun_ext_mean_mean)
    rename!(df, :T_ext_mean_std => :int_pun_ext_mean_std)
    rename!(df, :T_self_mean_mean => :int_pun_self_mean_mean)
    rename!(df, :T_self_mean_std => :int_pun_self_mean_std)
end

function process_simulation(
    input_dir::String,
    output_dir::String,
    process_function::Function;
    file_extension::String = ".csv",
)
    # Ensure the output directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Get a list of all files in the input directory matching the file extension
    files = filter(f -> endswith(f, file_extension), readdir(input_dir, join = true))

    for file in files
        # Load file
        data = read_simulation(file)

        # Process the data
        processed_data = process_function(data)

        # Create output file path
        output_file = joinpath(output_dir, basename(file))

        # Save processed data
        save_simulation(processed_data, output_file)
    end
end

end # module IOHandler
