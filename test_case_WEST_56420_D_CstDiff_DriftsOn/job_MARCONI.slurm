#!/bin/bash

# Nom de job
#SBATCH --job-name=dev_test

# Nom de projet
#SBATCH -A FUA38_TSVV6_24
#SBATCH -V

# Partition (queue) => skl_fua_dbg / skl_fua_prod
#SBATCH -p skl_fua_dbg

# Export environnement variables
#SBATCH --export=ALL

# Ressources
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
##SBATCH --mem=118000

# Wall time
#SBATCH --time=01:00:00

# Output file
#SBATCH --output=output.log

# To have a bit of verbose from the shell
set -vx

# Runs the job script file
cd $SLURM_SUBMIT_DIR

source run_case_MARCONI.sh
