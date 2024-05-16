#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

TRAINING_TYPE=$1
MODEL_TYPE=$2

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch

#on manivald
export RUNCMD="singularity exec -B /scratch/persistent --env PYTHONPATH=`pwd` --nv /home/software/singularity/pytorch.simg:2024-04-30 "

#on local system
# export RUNCMD="singularity exec --env PYTHONPATH=`pwd` --nv /home/joosep/HEP-KBFI/singularity/pytorch.simg "

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=$TRAINING_TYPE training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] model_type=$MODEL_TYPE
