# Installation

```
git clone git@github.com:Laurits7/ml-tau-en-reg.git
```

# Datasets

The latest merged ntuples for training are here:
```
$ du -csh /scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/*
490M	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/qq_test.parquet
2.0G	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/qq_train.parquet
30M	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/zh_test.parquet
119M	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/zh_train.parquet
26M	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/z_test.parquet
101M	/scratch/persistent/joosep/ml-tau/20240402_full_stats_merged/z_train.parquet
```

# Results
```
$ du -csh /local/joosep/ml-tau-en-reg/results/240611_PT2layers/*/*/*
7.5M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/dm_multiclass/LorentzNet
11M     /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/dm_multiclass/ParticleTransformer
7.0M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/dm_multiclass/SimpleDNN
7.2M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/jet_regression/LorentzNet
9.8M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/jet_regression/ParticleTransformer
6.8M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v1/jet_regression/SimpleDNN
7.5M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/dm_multiclass/LorentzNet
11M     /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/dm_multiclass/ParticleTransformer
7.0M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/dm_multiclass/SimpleDNN
7.2M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/jet_regression/LorentzNet
9.8M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/jet_regression/ParticleTransformer
6.7M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v2/jet_regression/SimpleDNN
7.5M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/dm_multiclass/LorentzNet
11M     /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/dm_multiclass/ParticleTransformer
7.0M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/dm_multiclass/SimpleDNN
7.2M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/jet_regression/LorentzNet
9.7M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/jet_regression/ParticleTransformer
6.6M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v3/jet_regression/SimpleDNN
7.5M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/dm_multiclass/LorentzNet
11M     /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/dm_multiclass/ParticleTransformer
7.0M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/dm_multiclass/SimpleDNN
7.1M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/jet_regression/LorentzNet
9.8M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/jet_regression/ParticleTransformer
6.7M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v4/jet_regression/SimpleDNN
7.5M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/dm_multiclass/LorentzNet
11M     /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/dm_multiclass/ParticleTransformer
7.0M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/dm_multiclass/SimpleDNN
7.4M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/jet_regression/LorentzNet
9.7M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/jet_regression/ParticleTransformer
6.8M    /local/joosep/ml-tau-en-reg/results/240611_PT2layers/v5/jet_regression/SimpleDNN
```

# Running

All the necessary packages are installed to the singularity image used in the ```run.sh``` script.
In order to run the scripts do:
```bash
./run.sh python3 [XYZ]
```


# Training

To test locally on a fraction of the data
```
./run.sh python3 enreg/scripts/trainModel.py output_dir=training-outputs/mytest fraction_train=0.1 fraction_valid=0.1 training.num_epochs=2 model_type=SimpleDNN training_type=jet_regression
./run.sh python3 enreg/scripts/trainModel.py output_dir=training-outputs/mytest fraction_train=0.1 fraction_valid=0.1 training.num_epochs=2 model_type=SimpleDNN training_type=dm_multiclass
```
The configuration the models starts at `enreg/config/model_training.yaml`.

To submit the training of the models to `gpu0`, check and run
```
./enreg/scripts/submit-pytorch-gpu-all.sh
```

# Plotting

Change `enreg/config/benchmarking.yaml` and `enreg/config/metrics/regression.yaml` as needed.

```
./run.sh python3 enreg/scripts/calculate_regression_metrics.py
```
and
```
notebooks/DM_CM.ipynb
notebooks/losses.ipynb
```

## Notebooks

```
./run.sh jupyter notebook --no-browser
```

# Creating the input data

## Simulation with Key4HEP

Edit `sim/run_sim.sh` to change the output directory, then generate 100 events as follows:
```
cd sim
./run_sim.sh 1 p8_ee_ZH_Htautau_ecm380.cmd 
```

## Creating ML ntuples

To produce the jet-based ML ntuples from the .root and .hepmc files 
```
./run.sh python3 enreg/scripts/ntupelize_edm4hep.py
bash /home/joosep/tmp/NA7VJ17OVH/executables/execute0.sh  
```

# Contributing

Contributing to the project has the following requirements:

- Document each functionality you add (i.e., docstrings for each function you add)
- Follow the PEP8 guidelines
- Create a new branch when making edits and create a Pull Request (PR) for it to be merged to the `main` branch. Direct pushes to the `main` branch are disabled.
- Recommended: Add unit tests for the functionality
