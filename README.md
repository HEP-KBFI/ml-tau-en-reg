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

# Running

All the necessary packages are installed to the singularity image used in the ```run.sh``` script.
In order to run the scripts do:
```bash
./run.sh python3 [XYZ]
```

# Training

To test locally on a fraction of the data
```
./run.sh python3 enreg/scripts/trainModel.py output_dir= model_type=SimpleDNN training_type=jet_regression fraction_train=0.1 fraction_valid=0.1 training.num_epochs=2
./run.sh python3 enreg/scripts/trainModel.py output_dir= model_type=SimpleDNN training_type=dm_multiclass fraction_train=0.1 fraction_valid=0.1 training.num_epochs=2
```
The configuration the models starts at `enreg/config/model_training.yaml`.

To submit the training of the models to gpu0, check and run
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

# Contributing

Contributing to the project has the following requirements:

- Document each functionality you add (i.e., docstrings for each function you add)
- Follow the PEP8 guidelines
- Create a new branch when making edits and create a Pull Request (PR) for it to be merged to the `main` branch. Direct pushes to the `main` branch are disabled.
- Recommended: Add unit tests for the functionality
