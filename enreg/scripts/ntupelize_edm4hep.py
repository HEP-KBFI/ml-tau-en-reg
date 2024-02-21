import os
import time
import glob
import hydra
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from enreg.tools.data_management import ntupelizer as nt



def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(ak.Record(data), output_path)


def process_single_file(
    input_path: str,
    output_dir: str,
    sample: str,
    cfg: DictConfig
):
    file_name = os.path.basename(input_path).replace(".root", ".parquet")
    output_ntuple_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_ntuple_path):
        start_time = time.time()
        remove_bkg = cfg.samples[sample].is_signal
        arrays = nt.load_single_file_contents(input_path, cfg.tree_path, cfg.branches)
        data = nt.process_input_file(arrays, remove_background=remove_bkg)
        save_record_to_file(data, output_ntuple_path)
        end_time = time.time()
        print(f"Finished processing in {end_time-start_time} s.")
    else:
        print("File already processed, skipping.")


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def process_all_input_files(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    for sample_name in cfg.samples_to_process:
        output_dir = cfg.samples[sample_name].output_dir
        input_dir = cfg.samples[sample_name].input_dir
        os.makedirs(output_dir, exist_ok=True)
        input_wcp = os.path.join(input_dir, "*.root")
        if cfg.test_run:
            n_files = 3
        else:
            n_files = None
        input_paths = glob.glob(input_wcp)[:n_files]
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=10)
            pool.starmap(process_single_file, zip(input_paths, repeat(output_dir), repeat(sample_name), repeat(cfg)))
        else:
            for path in input_paths:
                process_single_file(path, output_dir, sample_name, cfg)


if __name__ == "__main__":
    process_all_input_files()