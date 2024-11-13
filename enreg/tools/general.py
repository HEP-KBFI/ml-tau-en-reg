import os
import glob
import json
import vector
import numpy as np
import awkward as ak


# def load_all_data(input_loc: str | list, n_files: int = None, columns: list = None) -> ak.Array:
def load_all_data(input_loc, n_files: int = None, columns: list = None) -> ak.Array:

    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the files
    are located or a wildcard path.

    Args:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns will
            be loaded

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_files = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_files = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_files = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_files = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc")
    else:
        raise ValueError(f"Unexpected input_loc")
    input_data = []
    # for file_path in tqdm.tqdm(sorted(input_files)):
    for i, file_path in enumerate(sorted(input_files)):
        print(f"[{i+1}/{len(input_files)}] Loading from {file_path}")
        try:
            input_data.append(load_parquet(file_path, columns=columns))
        except ValueError:
            print(f"{file_path} does not exist")
    if len(input_data) > 0:
        data = ak.concatenate(input_data)
        print("Input data loaded")
    else:
        raise ValueError(f"No files found in {input_loc}")
    return data


def load_parquet(input_path: str, columns: list = None) -> ak.Array:
    """ Loads the contents of the .parquet file specified by the input_path

    Args:
        input_path : str
            The path to the .parquet file to be loaded.
        columns : list
            Names of the columns/branches to be loaded from the .parquet file

    Returns:
        input_data : ak.Array
            The data from the .parquet file
    """
    ret = ak.from_parquet(input_path, columns=columns)
    ret = ak.Array({k: ret[k] for k in ret.fields})
    return ret


def get_decaymode(pdg_ids):
    """Tau decaymodes are the following:
    decay_mode_mapping = {
        0: 'OneProng0PiZero',
        1: 'OneProng1PiZero',
        2: 'OneProng2PiZero',
        3: 'OneProng3PiZero',
        4: 'OneProngNPiZero',
        5: 'TwoProng0PiZero',
        6: 'TwoProng1PiZero',
        7: 'TwoProng2PiZero',
        8: 'TwoProng3PiZero',
        9: 'TwoProngNPiZero',
        10: 'ThreeProng0PiZero',
        11: 'ThreeProng1PiZero',
        12: 'ThreeProng2PiZero',
        13: 'ThreeProng3PiZero',
        14: 'ThreeProngNPiZero',
        15: 'RareDecayMode'
        16: 'LeptonicDecay'
    }
    0: [0, 5, 10]
    1: [1, 6, 11]
    2: [2, 3, 4, 7, 8, 9, 12, 13, 14, 15]
    """
    pdg_ids = np.abs(np.array(pdg_ids))
    unique, counts = np.unique(pdg_ids, return_counts=True)
    common_particles = [16, 130, 211, 13, 14, 12, 11]
    n_uncommon = len(set(unique) - set(common_particles))
    p_counts = {i: 0 for i in common_particles}
    p_counts.update(dict(zip(unique, counts)))
    if n_uncommon > 0:
        return 15
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 0:
        return 0
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 1:
        return 1
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 2:
        return 2
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 3:
        return 3
    elif np.sum(p_counts[211]) == 1 and p_counts[130] > 3:
        return 4
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 0:
        return 5
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 1:
        return 6
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 2:
        return 7
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 3:
        return 8
    elif np.sum(p_counts[211]) == 2 and p_counts[130] > 3:
        return 9
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 0:
        return 10
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 1:
        return 11
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 2:
        return 12
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 3:
        return 13
    elif np.sum(p_counts[211]) == 3 and p_counts[130] > 3:
        return 14
    elif np.sum(p_counts[11] + p_counts[13]) > 0:
        return 16
    else:
        return 15


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
    target_mapping = {
        -1: 15,  # As we are running DM classification only on signal sample, then HPS_dm of -1 = 15 (Rare)
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 10,
        6: 11,
        7: 11,
        8: 11,
        9: 11,
        10: 10,
        11: 11,
        12: 11,
        13: 11,
        14: 11,
        15: 15,
        16: 16,
    }
    return np.vectorize(target_mapping.get)(decaymodes)


def reinitialize_p4(p4_obj: ak.Array):
    """ Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta.
    """
    if "tau" in p4_obj.fields:
        p4 = vector.awk(
            ak.zip(
                {
                    "mass": p4_obj.tau,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    else:
        p4 = vector.awk(
            ak.zip(
                {
                    "energy": p4_obj.t,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    return p4


def load_json(path):
    """ Loads the contents of the .json file with the given path

    Args:
        path : str
            The location of the .json file

    Returns:
        data : dict
            The content of the loaded json file.
    """
    with open(path, "rt") as in_file:
        data = json.load(in_file)
    return data


class NpEncoder(json.JSONEncoder):
    """ Class for encoding various objects such that they could be saved to a json file"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_to_json(data, output_path):
    """ Saves data to a .json file located at `output_path`

    Args:
        data : dict
            The data to be saved
        output_path : str
            Destonation of the .json file

    Returns:
        None
    """
    with open(output_path, "wt") as out_file:
        json.dump(data, out_file, indent=4, cls=NpEncoder)
