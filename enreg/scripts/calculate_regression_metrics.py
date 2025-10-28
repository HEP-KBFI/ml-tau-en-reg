import os
import hydra
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from enreg.tools import general as g
from enreg.tools.metrics import energy_regression as er

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


@hydra.main(config_path="../config", config_name="benchmarking", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(os.path.expandvars(cfg.output_dir), exist_ok=True)

    sample_data = {}
    sample_algo_data = {}
    for sample in cfg.comparison_samples:
        base_ntuple_data = g.load_all_data(
            [str(os.path.join(cfg.base_ntuple_path, sample + ".parquet"))]
        )
        gen_jet_tau_vis_p4 = g.reinitialize_p4(base_ntuple_data["gen_jet_tau_p4s"])
        reco_jet_p4s = g.reinitialize_p4(base_ntuple_data.reco_jet_p4s)
        mask = ak.to_numpy((gen_jet_tau_vis_p4.energy > 1))
        base_ntuple_data = base_ntuple_data[mask]

        sample_data[sample] = base_ntuple_data

        algo_data = {}
        for algorithm, properties in cfg.metrics.regression.algorithms.items():
            if not properties.compare or properties.load_from_json:
                continue
            algo_output_data = g.load_all_data(
                [str(os.path.join(properties.ntuples_dir, sample + ".parquet"))]
            )

            # reconstructed tau pt from the model prediction
            # pred = log(gentau.pt/recojet.pt) -> gentau.pt = exp(pred) * recojet.pt
            tau_pt = (
                np.exp(algo_output_data["jet_regression"]["pred"]) * reco_jet_p4s.pt
            )

            algo_data[algorithm] = ak.to_numpy(tau_pt[mask])
        algo_data["RecoJet"] = ak.to_numpy(reco_jet_p4s.pt)[mask]
        sample_algo_data[sample] = ak.Record(algo_data)
    er.plot_energy_regression(sample_data, sample_algo_data, cfg)


if __name__ == "__main__":
    main()
