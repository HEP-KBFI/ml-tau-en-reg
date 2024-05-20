#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch

apptainer exec -B /home/laurits,/scratch/persistent,/local/joosep --env PYTHONPATH=`pwd` --nv /home/software/singularity/pytorch.simg\:2024-04-30 "$@"
