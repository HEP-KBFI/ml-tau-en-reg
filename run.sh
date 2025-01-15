#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
apptainer exec -B /scratch/persistent,/local,/home/laurits/ml-tau-en-reg/training-outputs,/home/hardiveski/ml-tau-en-reg --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha --nv /home/software/singularity/pytorch.simg\:2024-08-02 "$@"
