#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=20G
#SBATCH -o slurm-%x-%j-%N.out

#get commandline arguments
TRAINING_TYPE=$1
MODEL_TYPE=$2
TRAIN_SAMPS=$3
TEST_SAMPS=$4

export KERAS_BACKEND=torch

./run.sh python3 enreg/scripts/trainModel.py "@$"
