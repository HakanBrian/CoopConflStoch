#!/bin/bash
#SBATCH --time=06:00:00             # Time limit for the job (REQUIRED).
#SBATCH --job-name=CoopConflGame    # Job name
#SBATCH --exclusive                 # Allocate all cores in node.
#SBATCH --partition=normal          # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err             # Error file for this job.
#SBATCH -o slurm-%j.out             # Output file for this job.
#SBATCH -A <your project account>     # Project allocation account name (REQUIRED)
#SBATCH --mail-type ALL             # Send email when job starts/ends
 
module purge                      # Unload other software modules

# Execute the Julia script
julia CoopConflGameSLURM.jl