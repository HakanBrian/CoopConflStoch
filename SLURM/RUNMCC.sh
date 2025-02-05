#!/bin/bash
#SBATCH --time=1-00:00:00           # Time limit for the job (REQUIRED).
#SBATCH --job-name=CoopConflGame    # Job name
#SBATCH --nodes=1                   # Number of nodes to allocate. Same as SBATCH -N (Don't use this option for mpi jobs)
#SBATCH --exclusive                 # Allocate all cores in node.
#SBATCH --partition=normal          # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err             # Error file for this job.
#SBATCH -o slurm-%j.out             # Output file for this job.
#SBATCH -A <your project account>   # Project allocation account name (REQUIRED)
#SBATCH --mail-type ALL             # Send email when job starts/ends
#SBATCH --mail-user <your email>    # Email address to send email to
 
module purge                        # Unload other software modules

# Expand SLURM node list and format it for Julia
NODELIST=$(scontrol show hostnames $SLURM_NODELIST | awk '{print "(\"" $1 "\", :auto)"}' | paste -sd "," -)

# Run Julia with distributed processing
julia -e "using Distributed; addprocs([$NODELIST]); include(\"SLURM.jl\")"