#!/bin/bash

export OUTPUT_DIR=/home/$USER/ml-tau-en-reg/training-outputs/240516_fullstats/
export DATA_PATH=/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged

#for regression and decay mode, use only signal (tau) jets
export TRAIN_SAMPS=zh_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression LorentzNet
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression ParticleTransformer
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression SimpleDNN

sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass LorentzNet
sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass ParticleTransformer
sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass SimpleDNN

#for binary classification, use signal (tau) and background (non-tau) jets
export TRAIN_SAMPS=zh_train.parquet,qq_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification LorentzNet
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification ParticleTransformer
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification SimpleDNN
