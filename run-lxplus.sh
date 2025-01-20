#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
apptainer exec -B /eos/user/l/ltani --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha --nv /eos/user/l/ltani/singularity/pytorch_w_geom.simg "$@"
