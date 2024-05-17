#!/bin/bash

#for regression and decay mode, use only signal (tau) jets
export TRAIN_SAMPS=zh_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression LorentzNet $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression ParticleTransformer $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh jet_regression SimpleDNN $TRAIN_SAMPS $TEST_SAMPS

sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass LorentzNet $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass ParticleTransformer $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh dm_multiclass SimpleDNN $TRAIN_SAMPS $TEST_SAMPS

##for binary classification, use signal (tau) and background (non-tau) jets
export TRAIN_SAMPS=zh_train.parquet,qq_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification LorentzNet $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification ParticleTransformer $TRAIN_SAMPS $TEST_SAMPS
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification SimpleDNN $TRAIN_SAMPS $TEST_SAMPS
