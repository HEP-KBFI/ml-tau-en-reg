#!/bin/bash

sbatch enreg/scripts/train-pytorch-gpu-taus.sh jet_regression LorentzNet
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-taus.sh jet_regression ParticleTransformer
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-taus.sh jet_regression SimpleDNN
sleep 10

sbatch enreg/scripts/train-pytorch-gpu-taus.sh dm_multiclass LorentzNet
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-taus.sh dm_multiclass ParticleTransformer
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-taus.sh dm_multiclass SimpleDNN
sleep 10

sbatch enreg/scripts/train-pytorch-gpu-full.sh  binary_classification LorentzNet
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification ParticleTransformer
sleep 10
sbatch enreg/scripts/train-pytorch-gpu-full.sh binary_classification SimpleDNN
sleep 10
