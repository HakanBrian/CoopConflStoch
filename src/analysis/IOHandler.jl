module IOHandler

export save_simulation,
    read_simulation,
    read_matching_simulations,
    generate_filename_suffix,
    modify_filename,
    process_simulation

using ..MainSimulation.SimulationParameters
import ..MainSimulation.SimulationParameters: SimulationParameter

using CSV, DataFrames, Glob

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

function read_matching_simulations(
    filepath::String;
    pattern_template::String,
    extract_keys::Vector{String},
)
    # Extract the lowest folder in the filepath
    dir_path = dirname(filepath)
    folder_name = splitpath(dir_path)[end]  # Get the last folder in the path

    # Ensure the directory exists
    if !isdir(dir_path)
        error("Error: Directory '$dir_path' does not exist.")
    end

    # Convert pattern_template to glob pattern
    glob_pattern = replace(pattern_template, r"\{(\w+)\}" => "*")

    # Find matching files
    files = glob(glob_pattern, dir_path)

    # Check if files were found
    if isempty(files)
        error(
            "Error: No matching files found in '$dir_path' using pattern '$glob_pattern'.",
        )
    end

    println("Loading files from: $dir_path")

    # Load matching files into a dictionary
    simulations = Dict{Tuple{Vararg{String}},DataFrame}()

    for file in files
        filename = splitdir(file)[2]  # Extract filename from full path

        # Extract values based on keys in extract_keys
        key_values = []
        for key in extract_keys
            # Dynamically adjust regex based on the key
            pattern = if key == "punishment"
                Regex("$(folder_name)_([^_]+)")  # Replace "punishment" dynamically
            else
                Regex("$key=([^_]+)")  # General case for key=value format
            end

            m = match(pattern, filename)
            if m === nothing
                error("Error: Could not extract '$key' from filename '$filename'.")
            end
            push!(key_values, m.captures[1])  # Store extracted value
        end

        # Store DataFrame using tuple of extracted values as key
        simulations[Tuple(key_values)] = CSV.read(file, DataFrame)
    end

    return simulations
end

function generate_filename_suffix(
    param_dict::Dict{<:Any,<:Any},
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
