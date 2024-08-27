#!/bin/bash


#for regression and decay mode, use only signal (tau) jets
export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
export CLS_TRAIN_SAMPS=z_train.parquet,qq_train.parquet
export CLS_TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet

export BASE_DIR=training-outputs/Trainings


for i in `seq 1 1`; do
    export OUTDIR=$BASE_DIR/v$i
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT

    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=LorentzNet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT

    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=LorentzNet dataset
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=DeepSet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT
done
