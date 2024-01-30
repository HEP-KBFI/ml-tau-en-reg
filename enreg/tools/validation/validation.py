import os
import hydra
import vector
import numpy as np
import awkward as ak
import plotting as pl
from omegaconf import DictConfig
from ..general import load_all_data


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def validate_ntuples(cfg: DictConfig) -> None:
    output_dir = os.path.expandvars(cfg.validation.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for sample in cfg.validation.validation_samples:
        ntuple_dir = cfg.samples[sample].output_dir
        data = load_all_data(ntuple_dir)
        sample_output_dir = os.path.join(output_dir, sample)
        os.makedirs(sample_output_dir, exist_ok=True)
        jet_en_cm_path = os.path.join(sample_output_dir, "jet_energy_reco.png")
        plot_reco_jet_energy_cm(data, jet_en_cm_path)
        jet_en_cm_path = os.path.join(sample_output_dir, "cand_energy_reco.png")
        plot_reco_vs_gen_cand_energy_cm(data, jet_en_cm_path)
        plot_reco_cand_properties(data, ["pt", "px", "py", "pz"], sample_output_dir)


def plot_reco_jet_energy_cm(data, output_path):
    gen_jet_en_ = data.gen_jet_tau_vis_energy
    reco_jet_en = vector.awk(
        ak.zip(
            {
                "mass": data.reco_jet_p4s.tau,
                "x": data.reco_jet_p4s.x,
                "y": data.reco_jet_p4s.y,
                "z": data.reco_jet_p4s.z,
            }
        )
    ).energy
    gen_jet_en = (gen_jet_en_[gen_jet_en_ != -1]).to_numpy()
    reco_jet_en = (reco_jet_en[gen_jet_en_ != -1]).to_numpy()
    pl.plot_regression_confusion_matrix(
        y_true=gen_jet_en,
        y_pred=reco_jet_en,
        output_path=output_path,
        left_bin_edge=np.min([reco_jet_en, gen_jet_en]),
        right_bin_edge=np.max([reco_jet_en, gen_jet_en]),
        y_label="Reconstructed jet energy",
        x_label="Gen visible tau energy",
    )


def plot_reco_vs_gen_cand_energy_cm(data, output_path):
    gen_energy = data.reco_cand_matched_gen_energy
    reco_cand_energy = vector.awk(
        ak.zip(
            {
                "mass": data.reco_cand_p4s.tau,
                "x": data.reco_cand_p4s.x,
                "y": data.reco_cand_p4s.y,
                "z": data.reco_cand_p4s.z,
            }
        )
    ).energy
    gen_energy_ = ak.flatten((gen_energy[gen_energy != -1]), axis=1).to_numpy()
    reco_cand_energy_ = ak.flatten((reco_cand_energy[gen_energy != -1]), axis=1).to_numpy()
    pl.plot_regression_confusion_matrix(
        y_true=gen_energy_,
        y_pred=reco_cand_energy_,
        output_path=output_path,
        left_bin_edge=np.min([reco_cand_energy_, gen_energy_]),
        right_bin_edge=np.max([reco_cand_energy_, gen_energy_]),
        y_label="Reco candidate energy",
        x_label="Matched gen energy",
    )


def plot_reco_cand_properties(data, properties: list, output_dir: str):
    reco_cand_p4 = vector.awk(
        ak.zip(
            {
                "mass": data.reco_cand_p4s.tau,
                "x": data.reco_cand_p4s.x,
                "y": data.reco_cand_p4s.y,
                "z": data.reco_cand_p4s.z,
            }
        )
    )
    for property_ in properties:
        entries = ak.flatten(getattr(reco_cand_p4, property_), axis=-1)
        output_path = os.path.join(output_dir, f"{property_}.png")
        pl.plot_histogram(
            entries=entries,
            output_path=output_path,
            left_bin_edge=min(entries),
            right_bin_edge=max(entries),
            title=property_,
        )


if __name__ == "__main__":
    validate_ntuples()
