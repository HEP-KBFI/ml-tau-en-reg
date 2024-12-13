import os
import glob
import tqdm
import uproot
import hydra
import awkward as ak
from omegaconf import DictConfig


def load_root_file(path: str, tree_path: str = "tree", branches: list = None) -> ak.Array:
    """ Loads the CEPC dataset .root file.

    Parameters:
        path : str
            Path to the .root file
        tree_path : str
            Path in the tree in the .root file.
        branches : list
            [default: None] Branches to be loaded from the .root file. By default, all branches will be loaded.

    Returns:
        array : ak.Array
            Awkward array containing the .root file data
    """
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        arrays = tree.arrays(branches)
    return arrays


def save_parquet_file(data: ak.Array, output_path: str, verbosity: int = 0) -> None:
    if verbosity > 0:
        print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path, row_group_size=1024)


@hydra.main(config_path="../config", config_name="jetclass_parquetizer", version_base=None)
def main(cfg: DictConfig):
    for dataset, dataset_name in cfg.dataset_dirs.items():
        print(f"Processing {dataset} dataset")
        files_wcp = os.path.join(cfg.jetclass_root_dir, dataset_name, "*.root")
        dataset_parquet_dir = os.path.join(cfg.jetclass_parquet_dir, dataset)
        os.makedirs(dataset_parquet_dir, exist_ok=True)
        all_paths = glob.glob(files_wcp)
        for idx, path in tqdm.tqdm(enumerate(all_paths), total=len(all_paths)):
            arrays = load_root_file(path=path, branches=cfg.branches_of_interest)
            filename = os.path.basename(path).split(".")[0]
            output_path = os.path.join(dataset_parquet_dir, f"{filename}.parquet")
            save_parquet_file(data=arrays, output_path=output_path, verbosity=cfg.verbosity)

if __name__ == "__main__":
    main()
