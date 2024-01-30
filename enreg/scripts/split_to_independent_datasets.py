import os
import yaml
import glob
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="ml_datasets", version_base=None)
def split_to_datasets(cfg: DictConfig) -> None:
    list_dir = os.path.expandvars(cfg.list_dir)
    datasets_l = ["train", "test", "validation"]
    if cfg.only_append_to_test:
        datasets = {}
        all_previous_samples = []
        all_sample_paths = []
        for sample in cfg.samples_to_process:
            all_sample_paths.extend(glob.glob(os.path.join(cfg.samples[sample].output_dir, "*.parquet")))
        for dataset in datasets_l:
            output_path = os.path.join(list_dir, f"{dataset}.yaml")
            with open(output_path, "rt") as in_file:
                paths = list(yaml.safe_load(in_file)[dataset]["paths"])
                datasets[dataset] = paths
                all_previous_samples.extend(paths)
        new_samples = list(set(all_sample_paths) - set(all_previous_samples))
        datasets["test"].extend(new_samples)
        test_output_path = os.path.join(list_dir, "test.yaml")
        output_info = {"test": {"paths": datasets["test"]}}
        print(f"Outputting to: {output_path}")
        with open(test_output_path, "wt") as out_file:
            yaml.dump(output_info, out_file)
    else:
        total = sum([cfg[dataset] for dataset in datasets_l])
        fractions = {dataset: cfg[dataset] / total for dataset in datasets_l}
        datasets = {d: [] for d in datasets_l}
        for sample in cfg.samples_to_process:
            sample_paths = glob.glob(os.path.join(cfg.samples[sample].output_dir, "*.parquet"))
            n_files_in_sample = len(sample_paths)
            n_train_files = int(n_files_in_sample * fractions["train"])
            n_test_files = int(n_files_in_sample * fractions["test"])
            datasets["train"].extend(sample_paths[:n_train_files])
            datasets["test"].extend(sample_paths[n_train_files : n_train_files + n_test_files])
            datasets["validation"].extend(sample_paths[n_train_files + n_test_files :])
        for dataset in datasets:
            output_path = os.path.join(list_dir, f"{dataset}.yaml")
            output_info = {dataset: {"paths": datasets[dataset]}}
            print(f"Outputting to: {output_path}")
            with open(output_path, "wt") as out_file:
                yaml.dump(output_info, out_file)


if __name__ == "__main__":
    split_to_datasets()
