import os
import glob
import torch
import hydra
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from enreg.tools.data_management.particleTransformer_dataset import ParticleTransformerTauBuilder


def process_single_file(input_path: str, builder, output_path: str) -> None:
    if not os.path.exists(output_path):
        print("Opening file %s" % input_path)
        jets = ak.from_parquet(input_path)
        print("Processing jets...")
        pjets = builder.process_jets(jets)
        print("...done, writing output file %s" % output_path)
        merged_info = {field: jets[field] for field in jets.fields}
        merged_info.update(pjets)
        ak.to_parquet(ak.Record(merged_info), output_path)
    else:
        print("File already processed ... Skipping")


@hydra.main(config_path="../config", config_name="tau_builder", version_base=None)
def build_taus(cfg: DictConfig) -> None:
    print("<runBuilder>:")
    if cfg.builder == "ParticleTransformer":
        builder = ParticleTransformerTauBuilder(cfg=cfg.models.ParticleTransformer)
    builder.print_config()
    algo_output_dir = os.path.join(os.path.expandvars(cfg.output_dir), cfg.builder)
    input_paths = []
    output_paths = []
    for dataset in cfg.datasets_to_process:
        print(f"::: Collecting files {dataset} dataset :::")
        dataset_dir = os.path.join(algo_output_dir, dataset)
        for sample in cfg.samples_to_process:
            samples_dir = cfg.samples[sample].output_dir
            sample_output_dir = os.path.join(dataset_dir, sample)
            os.makedirs(
                os.path.join(sample_output_dir),
                exist_ok=True
            )
            if not os.path.exists(samples_dir):
                raise OSError("Ntuples do not exist: %s" % (samples_dir))
            if cfg.n_files == -1:
                n_files = None
            else:
                n_files = cfg.n_files
            sample_input_paths = glob.glob(os.path.join(samples_dir, "*.parquet"))[:n_files]
            sample_output_paths = [os.path.join(sample_output_dir, os.path.basename(ip)) for ip in sample_input_paths]
            print(f"\tFound {len(sample_input_paths)} input files for {sample} sample in the {dataset} dataset.")
            input_paths.extend(sample_input_paths)
            output_paths.extend(sample_output_paths)
    if cfg.use_multiprocessing:  # kuidas assignida õige output_dir sample 
        pool = multiprocessing.Pool(processes=12)
        pool.starmap(process_single_file, zip(input_paths, repeat(builder), output_paths))
    else:
        for input_path, output_path in zip(input_paths, output_paths):
            process_single_file(input_path=input_path, builder=builder, output_path=output_path)


if __name__ == "__main__":
    build_taus()
