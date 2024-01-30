#!/usr/bin/python3

import torch
import os
import glob
import hydra
import awkward as ak
import multiprocessing
from omegaconf import DictConfig
from itertools import repeat
# from ParticleTransformerTauBuilder import ParticleTransformerTauBuilder



def process_single_file(input_path: str, builder, output_dir) -> None:
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    if not os.path.exists(output_path):
        print("Opening file %s" % input_path)
        jets = ak.from_parquet(input_path)
        print("Processing jets...")
        pjets = builder.processJets(jets)
        print("...done, writing output file %s" % output_path)
        merged_info = {field: jets[field] for field in jets.fields if "grid" not in field}
        merged_info.update(pjets)
        ak.to_parquet(ak.Record(merged_info), output_path)
    else:
        print("File already processed ... Skipping")


@hydra.main(config_path="../config", config_name="tau_builder", version_base=None)
def build_taus(cfg: DictConfig) -> None:
    print(cfg)
    # print("<runBuilder>:")
    # if cfg.builder == "ParticleTransformer":
    #     builder = ParticleTransformerTauBuilder(verbosity=cfg.verbosity)
    # builder.printConfig()
    # algo_output_dir = os.path.join(os.path.expandvars(cfg.output_dir), cfg.builder)
    # sampletype = list(cfg.datasets["test"]["paths"])
    # for sample in cfg.samples_to_process:
    #     print("Processing sample %s" % sample)
    #     output_dir = os.path.join(algo_output_dir, sample)
    #     samples_dir = cfg.samples[sample].output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     if not os.path.exists(samples_dir):
    #         raise OSError("Ntuples do not exist: %s" % (samples_dir))
    #     if cfg.n_files == -1:
    #         n_files = None
    #     else:
    #         n_files = cfg.n_files
    #     if "parquet" in samples_dir:
    #         input_paths = [samples_dir]
    #         assert n_files == 1
    #     else:
    #         all_input_paths = glob.glob(os.path.join(samples_dir, "*.parquet"))
    #         if n_files is None:
    #             input_paths = all_input_paths
    #         else:
    #             input_paths = all_input_paths[cfg.start : cfg.start + n_files]
    #     if cfg.test_only:
    #         input_paths = [
    #             input_path
    #             for input_path in input_paths
    #             if os.path.basename(input_path) in [os.path.basename(sample) for sample in sampletype]
    #         ]
    #     print("Found %i input files." % len(input_paths))
    #     if cfg.use_multiprocessing:
    #         pool = multiprocessing.Pool(processes=12)
    #         pool.starmap(process_single_file, zip(input_paths, repeat(builder), repeat(output_dir)))
    #     else:
    #         for input_path in input_paths:
    #             process_single_file(input_path=input_path, builder=builder, output_dir=output_dir)


if __name__ == "__main__":
    build_taus()
