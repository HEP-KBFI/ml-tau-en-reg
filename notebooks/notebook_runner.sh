#!/bin/bash

echo "Running master evalutator notebook with parameters from evaluation_params.yaml"

../run.sh papermill master_evaluator.ipynb master_evaluator.ipynb -f evaluation_params.yaml