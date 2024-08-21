#!/bin/bash

if false; then
    #for regression and decay mode, use only signal (tau) jets
    export TRAIN_SAMPS=z_train.parquet
    export TEST_SAMPS=z_test.parquet,zh_test.parquet

    export OUTDIR=training-outputs/240626_train_on_z/v1

    # ==== JET_REGRESSION with SIMPLEDNN ====
    if false; then
        # First training with all the features
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats_lifetimes training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features,cand_lifetimes]

        # Second training with kinematics and features
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features]

        # Third training with only kinematics
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet dataset.feature_set=[cand_kinematics]
    fi

    # ==== DM_MULTICLASS with SIMPLEDNN ====
    if false; then
        # First training with all the features
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats_lifetimes training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features,cand_lifetimes]

        # Second training with kinematics and features
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features]

        # Third training with only kinematics
        sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet dataset.feature_set=[cand_kinematics]
    fi
fi

# ==== BINARY_CLASSIFICATION with SIMPLEDNN ====
if true; then
    export TRAIN_SAMPS=z_train.parquet,qq_train.parquet
    export TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet
    
    export OUTDIR=training-outputs/240626_train_on_z/v1
    
    # First training with all the features
    sbatch --mem-per-gpu 150G enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats_lifetimes training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features,cand_lifetimes]

    # Second training with kinematics and features
    sbatch --mem-per-gpu 150G enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin_feats training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=DeepSet dataset.feature_set=[cand_kinematics,cand_features]

    # Third training with only kinematics
    sbatch --mem-per-gpu 150G enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR/feats_kin training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=DeepSet dataset.feature_set=[cand_kinematics]
fi
