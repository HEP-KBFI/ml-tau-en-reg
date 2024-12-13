#!/bin/bash

export TRAIN_SAMPS=z_train.parquet
export TEST_SAMPS=z_test.parquet,zh_test.parquet
export CLS_TRAIN_SAMPS=z_train.parquet,qq_train.parquet
export CLS_TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet

export BASE_DIR=training-outputs/202041014_binary_classification
# TRAINING_SCRIPT=enreg/scripts/train-pytorch-gpu.sh
TRAINING_SCRIPT=enreg/scripts/submit-lumi-gpu.sh

for i in `seq 1 3`; do
    for trainSize in 2e3 1e4 1e5 1e6; do
        export OUTDIR=$BASE_DIR/v$i/trainfrac_$trainSize

        # -----------------------------------------------------------------------------------------------------------------
        # Jet regression

        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer comet.experiment=ParT_jr_$trainSize
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniDeepSet

        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_jr_$trainSize
        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_jr_$trainSize
        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_jr_$trainSize



        # -----------------------------------------------------------------------------------------------------------------
        # DM multiclass

        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=LorentzNet
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=ParticleTransformer comet.experiment=ParT_dm_$trainSize
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet

        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_dm_$trainSize
        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_dm_$trainSize
        sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_dm_$trainSize


        # -----------------------------------------------------------------------------------------------------------------
        # Binary classification

        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=LorentzNet comet.experiment=LorentzNet_bc_$trainSize
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=ParticleTransformer comet.experiment=ParT_bc_$trainSize
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=DeepSet comet.experiment=DeepSet_bc_$trainSize
        # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet comet.experiment=OmniDeepSet_bc_$trainSize

       sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_bc_$trainSize
       sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_bc_$trainSize
       sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_bc_$trainSize
    done
done
