import os
import time
import glob
import hydra
import awkward as ak
import multiprocessing
import numpy as np
from itertools import repeat
from omegaconf import DictConfig
from enreg.tools.data_management import ntupelizer as nt
from enreg.tools.data_management import ntupelizer_slurm_tools as nst


def save_record_to_file(data: ak.Record, output_path: str) -> None:
    print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path)


#process a list of input files, merge into a single output file
def process_and_merge(
    input_paths: list[str],
    output_path: str,
    cfg: DictConfig
):
    sample = os.path.basename(os.path.dirname(output_path))
    datas = []
    if not os.path.exists(output_path):
        start_time = time.time()
        remove_bkg = cfg.samples[sample].is_signal
        for input_path in input_paths:
            print(f"processing {input_path}")
            data = nt.process_input_file(input_path, cfg.tree_path, cfg.branches, remove_background=remove_bkg)
            #Record -> Array to allow concat later on
            data = ak.Array({k: data[k] for k in data.fields}) 
            datas.append(data)

        #merge outputs from all inputs
        data = ak.concatenate(datas)
        save_record_to_file(data, output_path)

        end_time = time.time()
        print(f"Finished processing in {end_time-start_time:.2f} s.")
    else:
        print(f"File {output_path} already processed, skipping.")


#actually run the ntuplizer on slurm
def run_job(cfg: DictConfig):
    input_paths = []
    with open(cfg.input_file, 'rt') as inFile:
        for line in inFile:
            input_paths.append(line.strip('\n'))
    output_paths = []
    with open(cfg.output_file, 'rt') as inFile:
        for line in inFile:
            output_paths.append(line.strip('\n'))

    #must have exactly one output file
    assert(len(output_paths) == 1)
    output_path = output_paths[0]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    process_and_merge(input_paths, output_path, cfg)


#input preparation on interactive node
def prepare_inputs(cfg: DictConfig):
    all_input_paths = []
    all_output_paths = []

    for sample_name in cfg.samples_to_process:
        print(f"creating inputs for {sample_name}")
        output_dir = cfg.samples[sample_name].output_dir
        input_dir = cfg.samples[sample_name].input_dir
        os.makedirs(output_dir, exist_ok=True)
        input_wcp = os.path.join(input_dir, "*.root")
        
        #divide the input list into chunks of files_per_job
        #each chunk of N input files will yield exactly one output file 
        input_paths = sorted(list(glob.glob(input_wcp)[:cfg.n_files]))
        input_path_chunks = list(np.array_split(input_paths, len(input_paths)//cfg.files_per_job))
        print(f"found {len(input_paths)} files, {len(input_path_chunks)} chunks")
        output_paths = [os.path.join(output_dir, os.path.basename(chunk[0]).replace(".root", ".parquet")) for chunk in input_path_chunks]

        all_output_paths.extend(output_paths)
        all_input_paths.extend(input_path_chunks)

    #create executables for slurm
    nst.multipath_slurm_ntupelizer(all_input_paths, all_output_paths)

@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    if cfg.slurm_run:
        run_job(cfg)
    else:
        prepare_inputs(cfg)

if __name__ == "__main__":
    main()
