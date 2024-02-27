#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out


apptainer run --bind /scratch/persistent/laurits --nv /home/laurits/ml-tau-en-reg/p310/kookjamoos/ python /home/laurits/ml-tau-en-reg/enreg/scripts/trainParticleTransformer.py