#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=20G
#SBATCH -o slurm-%x-%j-%N.out

./run.sh python3 enreg/scripts/trainModel.py "$@"
