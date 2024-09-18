#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=HPS_processing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH -o HPS_logs/slurm-%x-%j-%N.out

./run.sh python3 enreg/scripts/HPS/process_HPS.py "$@"