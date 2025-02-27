module IOHandler

export output!, save_simulation, read_simulation, add_suffix_to_filepath, process_simulation

using ..Populations, CSV, FilePathsBase, DataFrames

function output!(outputs::DataFrame, t::Int64, pop::Population)
    N = pop.parameters.population_size

    # Determine the base row for the current generation
    if t == 1
        output_row_base = 1
    else
        output_row_base = (floor(Int64, t / pop.parameters.output_save_tick) - 1) * N + 1
    end

    # Calculate the range of rows to be updated
    output_rows = output_row_base:(output_row_base+N-1)

    # Update the DataFrame with batch assignment
    outputs.generation[output_rows] = fill(t, N)
    outputs.individual[output_rows] = 1:N
    outputs.action[output_rows] = pop.action
    outputs.a[output_rows] = pop.norm
    outputs.p[output_rows] = pop.ext_pun
    outputs.T_ext[output_rows] = pop.int_pun_ext
    outputs.T_self[output_rows] = pop.int_pun_self
    outputs.payoff[output_rows] = pop.payoff

    nothing
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

function add_suffix_to_filepath(filepath::String, key::String)
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
