#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

env | grep CUDA
nvidia-smi -L
# ./run.sh python3 enreg/scripts/train_jetclass.py "$@"
# ./run.sh python3 enreg/scripts/trainModel_JetClass.py "$@"
./run.sh python3 enreg/scripts/trainModel.py "$@"
