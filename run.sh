#!/bin/bash
# apptainer exec -B /scratch/persistent/laurits --env PYTHONPATH=`pwd` /home/software/singularity/pytorch.simg\:2024-02-13  "$@"
apptainer exec -B /home/laurits/,/scratch/persistent/laurits,/local/joosep --env PYTHONPATH=`pwd` /home/software/singularity/pytorch.simg\:2024-03-07  "$@"