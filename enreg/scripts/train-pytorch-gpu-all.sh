#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

OUTPUT_DIR=/home/$USER/ml-tau-en-reg/training-outputs/240515_fullstats/

#shared input folder
# DATA_PATH=/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged

#local input folder
DATA_PATH=/home/$USER/ml-tau-en-reg/data/20240402_full_stats_merged/

#on manivald
# export RUNCMD="singularity exec -B /scratch/persistent --env PYTHONPATH=`pwd` --nv /home/software/singularity/pytorch.simg:2024-04-30 "

#on local system
export RUNCMD="singularity exec --env PYTHONPATH=`pwd` --nv /home/joosep/HEP-KBFI/singularity/pytorch.simg "

TRAIN_SAMPS_SIG=zh_train.parquet
TRAIN_SAMPS_FULL=qq_train.parquet,zh_train.parquet
TEST_SAMPS_FULL=z_test.parquet,zh_test.parquet,qq_test.parquet

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass training_samples=[$TRAIN_SAMPS_SIG] model_type=LorentzNet
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression training_samples=[$TRAIN_SAMPS_SIG] model_type=LorentzNet
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification training_samples=[$TRAIN_SAMPS_FULL] test_samples=[$TEST_SAMPS_FULL] model_type=LorentzNet

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass training_samples=[$TRAIN_SAMPS_SIG] model_type=SimpleDNN
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression training_samples=[$TRAIN_SAMPS_SIG] model_type=SimpleDNN
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification training_samples=[$TRAIN_SAMPS_FULL] test_samples=[$TEST_SAMPS_FULL] model_type=SimpleDNN

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass training_samples=[$TRAIN_SAMPS_SIG] model_type=ParticleTransformer
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression training_samples=[$TRAIN_SAMPS_SIG] model_type=ParticleTransformer
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification training_samples=[$TRAIN_SAMPS_FULL] test_samples=[$TEST_SAMPS_FULL] model_type=ParticleTransformer