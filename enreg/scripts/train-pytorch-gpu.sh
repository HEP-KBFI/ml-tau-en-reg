#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 30G
#SBATCH -o logs/slurm-%x-%j-%N.out

env | grep CUDA
nvidia-smi -L
./run.sh python3 enreg/scripts/trainModel.py "$@"
