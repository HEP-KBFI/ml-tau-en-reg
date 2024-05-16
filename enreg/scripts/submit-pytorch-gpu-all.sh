#!/bin/bash

export OUTPUT_DIR=/home/$USER/ml-tau-en-reg/training-outputs/240515_fullstats/
export DATA_PATH=/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged

export TRAIN_SAMPS=zh_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression LorentzNet
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression ParticleTransformer
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression SimpleDNN
sleep 10

# sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass LorentzNet
# sleep 10
# sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass ParticleTransformer
# sleep 10
# sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass SimpleDNN
# sleep 10

# export TRAIN_SAMPS=zh_train.parquet,qq_train.parquet
# export TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet
# sbatch enreg/scripts/train-pytorch-gpu-full.sh  binary_classification LorentzNet
# sleep 10
# sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification ParticleTransformer
# sleep 10
# sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification SimpleDNN
# sleep 10
