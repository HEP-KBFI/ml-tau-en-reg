#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

OUTPUT_DIR=/home/$USER/ml-tau-en-reg/training-outputs/240515_fullstats/
# data_path=/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged
DATA_PATH=/home/$USER/ml-tau-en-reg/data/20240402_full_stats_merged/

# export RUNCMD="singularity exec -B /scratch/persistent --env PYTHONPATH=`pwd` --nv /home/software/singularity/pytorch.simg:2024-04-30 "
export RUNCMD="singularity exec --env PYTHONPATH=`pwd` --nv /home/joosep/HEP-KBFI/singularity/pytorch.simg "

SAMPS_FULL=qq.parquet,zh.parquet
SAMPS_SIG=zh.parquet

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass samples_to_use=[$SAMPS_SIG] model_type=LorentzNet
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression samples_to_use=[$SAMPS_SIG] model_type=LorentzNet
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification samples_to_use=[$SAMPS_FULL] model_type=LorentzNet

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass samples_to_use=[$SAMPS_SIG] model_type=SimpleDNN
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression samples_to_use=[$SAMPS_SIG] model_type=SimpleDNN
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification samples_to_use=[$SAMPS_FULL] model_type=SimpleDNN

$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=dm_multiclass samples_to_use=[$SAMPS_SIG] model_type=ParticleTransformer
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=jet_regression samples_to_use=[$SAMPS_SIG] model_type=ParticleTransformer
$RUNCMD python3 enreg/scripts/trainModel.py data_path=$DATA_PATH training_type=binary_classification samples_to_use=[$SAMPS_FULL] model_type=ParticleTransformer