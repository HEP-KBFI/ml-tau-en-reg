#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

TRAINING_TYPE=$1
MODEL_TYPE=$2

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch

OUTPUT_DIR=/home/$USER/ml-tau-en-reg/training-outputs/240515_fullstats/

#shared input folder, should be no need to modify
# DATA_PATH=/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged

#local input folder
DATA_PATH=/home/$USER/ml-tau-en-reg/data/20240402_full_stats_merged/

#on manivald
# export RUNCMD="singularity exec -B /scratch/persistent --env PYTHONPATH=`pwd` --nv /home/software/singularity/pytorch.simg:2024-04-30 "

#on local system
export RUNCMD="singularity exec --env PYTHONPATH=`pwd` --nv /home/joosep/HEP-KBFI/singularity/pytorch.simg "

#samples for binary classification, taus and non-taus
TRAIN_SAMPS=zh_train.parquet,qq_train.parquet
TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=$TRAINING_TYPE training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] model_type=$MODEL_TYPE
