#!/bin/bash


#for regression and decay mode, use only signal (tau) jets
export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
for i in `seq 1 1`; do
    export OUTDIR=training-outputs/240809_nw1/v$i
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=SimpleDNN

    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=LorentzNet
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=SimpleDNN
done

#for binary classification, use signal (tau) and background (non-tau) jets
export TRAIN_SAMPS=z_train.parquet,qq_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet
for i in `seq 1 1`; do
    export OUTDIR=training-outputs/240710_memopt/v$i
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=LorentzNet dataset.max_cands=32 training.batch_size=512
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=ParticleTransformer dataset.max_cands=32 training.batch_size=512
    sbatch enreg/scripts/train-pytorch-gpu.sh output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=binary_classification model_type=SimpleDNN dataset.max_cands=32 training.batch_size=512
done
