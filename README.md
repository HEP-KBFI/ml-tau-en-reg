# Installation

```
git clone git@github.com:Laurits7/ml-tau-en-reg.git
```

# Datasets

The latest merged ntuples for training are here:
```
$ du -csh /scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/*
390M	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/qq_test.parquet
1.6G	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/qq_train.parquet
93M	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/zh_test.parquet
370M	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/zh_train.parquet
78M	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/z_test.parquet
312M	/scratch/persistent/joosep/ml-tau/20240625_all_2M_merged/z_train.parquet
```

# Results
```
$ du -csh /local/joosep/ml-tau-en-reg/results/240625_all_2M/*/*/*
7.9M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/dm_multiclass/LorentzNet
11M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/dm_multiclass/ParticleTransformer
7.4M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/dm_multiclass/SimpleDNN
9.3M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/jet_regression/LorentzNet
12M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/jet_regression/ParticleTransformer
8.6M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v1/jet_regression/SimpleDNN
7.9M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/dm_multiclass/LorentzNet
11M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/dm_multiclass/ParticleTransformer
7.4M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/dm_multiclass/SimpleDNN
9.2M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/jet_regression/LorentzNet
12M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/jet_regression/ParticleTransformer
8.5M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v2/jet_regression/SimpleDNN
7.9M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/dm_multiclass/LorentzNet
11M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/dm_multiclass/ParticleTransformer
7.4M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/dm_multiclass/SimpleDNN
9.1M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/jet_regression/LorentzNet
12M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/jet_regression/ParticleTransformer
8.6M	/local/joosep/ml-tau-en-reg/results/240625_all_2M/v3/jet_regression/SimpleDNN
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
