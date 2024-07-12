#!/bin/bash

#for regression and decay mode, use only signal (tau) jets
export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet

export OUTDIR=training-outputs/240626_train_on_z/v1

# ==== JET_REGRESSION with SIMPLEDNN ====
# First training with all the features
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats_lifetimes training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=SimpleDNN dataset.feature_set=[cand_kinematics,cand_features,cand_lifetimes]

# Second training with kinematics and features
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=SimpleDNN dataset.feature_set=[cand_kinematics,cand_features]

# Third training with only kinematics
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=SimpleDNN dataset.feature_set=[cand_kinematics]


# ==== DM_MULTICLASS with SIMPLEDNN ====

# First training with all the features
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats_lifetimes training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=SimpleDNN dataset.feature_set=[cand_kinematics,cand_features,cand_lifetimes]

# Second training with kinematics and features
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=SimpleDNN dataset.feature_set=[cand_kinematics,cand_features]

# Third training with only kinematics
sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=SimpleDNN dataset.feature_set=[cand_kinematics]
