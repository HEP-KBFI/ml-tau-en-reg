import os
import hydra
import string
import random
import awkward as ak
import multiprocessing
from omegaconf import DictConfig
from enreg.tools.models import HPS
from enreg.tools import general as g

os.environ['NUMEXPR_MAX_THREADS'] = '3'
os.environ['NUMEXPR_NUM_THREADS'] = '3'

def generate_run_id(run_id_len=10):
    """ Creates a random alphanumeric string with a length of run_id_len

    Args:
        run_id_len : int
            [default: 10] Length of the alphanumeric string

    Returns:
        random_string : str
            The randomly generated alphanumeric string
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=run_id_len))


@hydra.main(config_path="../../config", config_name="tau_builder", version_base=None)
def main(cfg: DictConfig) -> None:
    files = [os.path.join(cfg.base_ntuple_dir, file) for file in cfg.files]
    data = g.load_all_data(files)
    builder = HPS.HPSTauBuilder(cfg=cfg.models.HPS)
    processed_data = builder.process_jets(data)
    pred_pt = g.reinitialize_p4(processed_data['tau_p4s']).pt
    true_pt = g.reinitialize_p4(data.gen_jet_tau_p4s).pt
    
    data_to_save = {
        "pred_decay_mode": processed_data['tau_decaymode'],
        "true_decay_mode": data.gen_jet_tau_decaymode,
        "pred_pt": pred_pt,
        "true_pt": true_pt,
    }
    random_file_name = f"{generate_run_id(10)}.parquet"
    os.makedirs(cfg.output_dir, exist_ok=True)
    ak.to_parquet(ak.Record(data_to_save), os.path.join(cfg.output_dir, random_file_name))


if __name__ == "__main__":
    main()
