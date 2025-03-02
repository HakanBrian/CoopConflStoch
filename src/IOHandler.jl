module IOHandler

export save_simulation, read_simulation, generate_filename_suffix, modify_filename, process_simulation

using CSV, DataFrames

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
        error("Error: Directory '$dir_path' does not exist. Please create it before saving.")
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

function generate_filename_suffix(param_dict::Dict{Symbol, <:Number}, condition::String="Filtered"; time_point::Union{Nothing, Int}=nothing)
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
