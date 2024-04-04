import os
import glob
import time
import hydra
import shutil
import awkward as ak
from omegaconf import DictConfig
from enreg.tools.data_management import ntupelizer as nt


tmp_dir = "/home/laurits/tmp/EEEPRGGQTR"

def prepare_resubmission(tmp_dir):
    resubmission_dir = os.path.join(tmp_dir, "executables", "resubmit")
    shutil.rmtree(resubmission_dir)
    os.makedirs(resubmission_dir, exist_ok=True)
    input_paths_files = []
    output_paths_files = []
    for path in glob.glob(os.path.join(tmp_dir, "error_files", "*")):
        if os.path.getsize(path) != 0:
            index = os.path.basename(path).strip("error")
            print(path)
            executable_path = os.path.join(os.path.join(tmp_dir, "executables", f"execute{index}.sh"))
            new_executable_path = os.path.join(os.path.join(resubmission_dir, f"execute{index}.sh"))
            input_paths_file = os.path.join(tmp_dir, f"input_paths_{index}.txt")
            output_paths_file = os.path.join(tmp_dir, f"output_paths_{index}.txt")
            input_paths_files.append(input_paths_file)
            output_paths_files.append(output_paths_file)
            # print(f"Copying {executable_path} to {new_executable_path}")
            shutil.copy(executable_path, new_executable_path)
    print(f"Jobs to resubmit: {len(input_paths_files)}")
    print(f"Run `bash enreg/scripts/submit_builder_batchJobs.sh {resubmission_dir}`")
    return input_paths_files, output_paths_files


def find_faulty_files(input_paths_files, output_paths_files):
    input_file_list = []
    output_file_list = []
    for input_paths_file, output_paths_file in zip(input_paths_files, output_paths_files):
        with open(input_paths_file, "rt") as inFile:
            for line in inFile:
                input_file_list.append(line.strip('\n'))
        with open(output_paths_file, "rt") as inFile:
            for line in inFile:
                output_file_list.append(line.strip('\n'))
    return input_file_list, output_file_list


def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(data, output_path)


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
def main(cfg: DictConfig) -> None:
    input_paths_files, output_paths_files = prepare_resubmission(cfg.tmp_dir)
    input_file_list, output_file_list = find_faulty_files(input_paths_files, output_paths_files)
    fails = []
    for i, (input_file, output_file) in enumerate(zip(input_file_list, output_file_list)):
        print(f"{i}/{len(output_file_list)}")
        try:
            process_single_file(input_file, output_file, cfg)
        except Exception as e:
            print("-------------------------------")
            print(e)
            print(f"Failed to process {input_file}")
            fails.append(input_file)
            print("-------------------------------")
    print(f"Number bad files: {len(fails)}")


if __name__ == "__main__":
    main()