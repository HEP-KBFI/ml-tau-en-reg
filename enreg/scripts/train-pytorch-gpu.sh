#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
TRAINING_SCRIPT=src/endtoend_simple.py
cd ~/ml-tau-reco

singularity exec -B /scratch/persistent --nv $IMG python3 $TRAINING_SCRIPT
