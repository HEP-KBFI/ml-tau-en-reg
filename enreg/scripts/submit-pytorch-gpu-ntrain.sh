#!/bin/bash

export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet

for trainfrac in 0.001 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    export OUTDIR=training-outputs/240819_omnideepset/trainfrac_$trainfrac
    sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
    sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=SimpleDNN
    sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer
    sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT
    sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniDeepSet
done
