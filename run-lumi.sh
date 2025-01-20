#!/bin/bash

cd /scratch/project_465001293/ml-tau-en-reg

# module load LUMI/24.03 partition/G

export IMG=/scratch/project_465001293/2024-10-04_pytorch_rocm.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465001293/tensorflow_datasets
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1  # to suppress ROCm fusion is enabled messages
export ROCM_PATH=/opt/rocm
export KERAS_BACKEND=torch
#export MIOPEN_DISABLE_CACHE=true
#export NCCL_DEBUG=INFO
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4

# env
#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465001293 \
    -B /tmp \
    --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    --env CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
     $IMG "$@"
