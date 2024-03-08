import os
import time
import glob
import hydra
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from enreg.tools.data_management import ntupelizer as nt
from enreg.tools.data_management import ntupelizer_slurm_tools as nst



def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(ak.Record(data), output_path)


def process_single_file(
    input_path: str,
    output_path: str,
    cfg: DictConfig
):
    sample = os.path.basename(os.path.dirname(output_path))
    if not os.path.exists(output_path):
        start_time = time.time()
        remove_bkg = cfg.samples[sample].is_signal
        data = nt.process_input_file(input_path, cfg.tree_path, cfg.branches, remove_background=remove_bkg)
        save_record_to_file(data, output_path)
        end_time = time.time()
        print(f"Finished processing in {end_time-start_time} s.")
    else:
        print("File already processed, skipping.")


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def process_all_input_files(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
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
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            process_single_file(input_path, output_path, cfg)
    else:
        all_input_paths = []
        all_output_paths = []
        for sample_name in cfg.samples_to_process:
            print(sample_name)
            output_dir = cfg.samples[sample_name].output_dir
            input_dir = cfg.samples[sample_name].input_dir
            os.makedirs(output_dir, exist_ok=True)
            input_wcp = os.path.join(input_dir, "*.root")
            if cfg.test_run:
                n_files = 3
            else:
                n_files = None
            input_paths = glob.glob(input_wcp)
            file_names = [os.path.basename(input_path).replace(".root", ".parquet") for input_path in input_paths]
            output_paths = [os.path.join(output_dir, file_name) for file_name in file_names]
            all_output_paths.extend(output_paths)
            all_input_paths.extend(input_paths)
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=10)
            pool.starmap(process_single_file, zip(all_input_paths, all_output_paths, repeat(cfg)))
        elif cfg.use_slurm:
            nst.multipath_slurm_tau_builder(all_input_paths, all_output_paths)
        else:
            for input_path, output_path in zip(all_input_paths, all_output_paths):
                process_single_file(input_path, output_path, cfg)


if __name__ == "__main__":
    process_all_input_files()