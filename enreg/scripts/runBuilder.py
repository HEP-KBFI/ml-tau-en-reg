import os
import glob
import hydra
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from enreg.tools.data_management.particleTransformer_dataset import ParticleTransformerTauBuilder
from enreg.tools.models.HPS import HPSTauBuilder
from enreg.tools.data_management import tau_builder_tools as tbt


@hydra.main(config_path="../config", config_name="tau_builder", version_base=None)
def build_taus(cfg: DictConfig) -> None:
    print("<runBuilder>:")
    algo_output_dir = os.path.join(os.path.expandvars(cfg.output_dir), cfg.builder)
    if cfg.builder == "ParticleTransformer":
        builder = ParticleTransformerTauBuilder(cfg=cfg.models.ParticleTransformer)
    elif cfg.builder == "HPS":
        builder = HPSTauBuilder(cfg=cfg.models.HPS)
    else:
        raise NotImplementedError(f"Please implement the tau builder for [{cfg.builder}] algorithm")
    builder.print_config()
    if cfg.slurm_run:
        input_paths = []
        with open(cfg.input_file, 'rt') as inFile:
            for line in inFile:
                input_paths.append(line.strip('\n'))
        output_paths = []
        with open(cfg.output_file, 'rt') as inFile:
            for line in inFile:
                output_paths.append(line.strip('\n'))
        for input_path, output_path in zip(input_paths, output_paths):
            tbt.process_single_file(input_path, builder, output_path)
    else:
        input_paths = []
        output_paths = []
        for dataset in cfg.datasets_to_process:
            print(f"::: Collecting files {dataset} dataset :::")
            dataset_dir = os.path.join(algo_output_dir, dataset)
            for sample in cfg.samples_to_process:
                samples_dir = os.path.join(cfg.PT_tauID_ntuple_dir, dataset, sample)
                sample_output_dir = os.path.join(dataset_dir, sample)
                os.makedirs(
                    os.path.join(sample_output_dir),
                    exist_ok=True
                )
                if not os.path.exists(samples_dir):
                    raise OSError(f"No ntuples found in {samples_dir}")
                if cfg.n_files == -1:
                    n_files = None
                else:
                    n_files = cfg.n_files
                sample_input_paths = glob.glob(os.path.join(samples_dir, "*.parquet"))[:n_files]
                sample_output_paths = [os.path.join(sample_output_dir, os.path.basename(ip)) for ip in sample_input_paths]
                print(f"\tFound {len(sample_input_paths)} input files for {sample} sample in the {dataset} dataset.")
                input_paths.extend(sample_input_paths)
                output_paths.extend(sample_output_paths)
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=13)
            pool.starmap(tbt.process_single_file, zip(input_paths, repeat(builder), output_paths))
        elif cfg.use_slurm:
            tbt.multipath_slurm_tau_builder(input_paths, output_paths)
        else:
            for input_path, output_path in zip(input_paths, output_paths):
                tbt.process_single_file(input_path=input_path, builder=builder, output_path=output_path)


if __name__ == "__main__":
    build_taus()
