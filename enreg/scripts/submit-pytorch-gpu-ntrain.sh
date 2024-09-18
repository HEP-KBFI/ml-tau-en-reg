#!/bin/bash

export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
export CLS_TRAIN_SAMPS=z_train.parquet,qq_train.parquet
export CLS_TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet

export BASE_DIR=training-outputs/240826_Long_comparison

for i in `seq 1 1`; do
    for trainfrac in 0.0008 0.008 0.08 0.8; do
        export OUTDIR=$BASE_DIR/v$i/trainfrac_$trainfrac
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer comet.experiment=ParT_jr_$trainfrac
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT comet.experiment=OmniParT_jr_$trainfrac
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniDeepSet

        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=LorentzNet
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=ParticleTransformer comet.experiment=ParT_dm_$trainfrac
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT comet.experiment=OmniParT_dm_$trainfrac
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet

        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=LorentzNet comet.experiment=LorentzNet_bc_$trainfrac
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=ParticleTransformer comet.experiment=ParT_bc_$trainfrac
        sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT comet.experiment=OmniParT_bc_$trainfrac
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=DeepSet comet.experiment=DeepSet_bc_$trainfrac
        # sbatch enreg/scripts/train-pytorch-gpu.sh fraction_train=$trainfrac output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet comet.experiment=OmniDeepSet_bc_$trainfrac

    done
done
