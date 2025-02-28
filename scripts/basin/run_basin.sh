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

# Expand SLURM node list and determine allocated CPUs per node
NODELIST=$(scontrol show hostnames $SLURM_NODELIST)

# Construct addprocs list with explicit CPU allocations
NODEINFO=$(while read -r node; do
    cpus=$(scontrol show node $node | awk '/CPUAlloc/ {print $1}' | cut -d= -f2)
    echo "(\"$node\", $cpus)"
done <<< "$NODELIST" | paste -sd "," -)

# Run Julia with distributed processing using allocated CPU counts
julia -e "using Distributed; addprocs([$NODEINFO]); include(\"generate_basin.jl\")"
