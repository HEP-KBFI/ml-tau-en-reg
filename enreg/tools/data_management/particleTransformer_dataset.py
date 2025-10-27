import torch
import math
import numpy as np
import awkward as ak
import enreg.tools.general as g
from omegaconf import DictConfig
from torch.utils.data import IterableDataset
import enreg.tools.data_management.features as f

from collections.abc import Sequence


def stack_and_pad_features(cand_features, max_cands):
    cand_features_tensors = np.stack(
        [
            ak.pad_none(cand_features[feat], max_cands, clip=True)
            for feat in cand_features.fields
        ],
        axis=-1,
    )
    cand_features_tensors = ak.to_numpy(ak.fill_none(cand_features_tensors, 0))
    # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
    cand_features_tensors = np.swapaxes(cand_features_tensors, 1, 2)

    cand_features_tensors[np.isnan(cand_features_tensors)] = 0
    cand_features_tensors[np.isinf(cand_features_tensors)] = 0
    return cand_features_tensors


def load_row_groups(filename):
    metadata = ak.metadata_from_parquet(filename)
    num_row_groups = metadata["num_row_groups"]
    col_counts = metadata["col_counts"]
    return [
        RowGroup(filename, row_group, col_counts[row_group])
        for row_group in range(num_row_groups)
    ]


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


class ParticleTransformerDataset(IterableDataset):
    def __init__(
        self, row_groups: Sequence[RowGroup], cfg: DictConfig, reco_jet_pt_cut: float
    ):
        self.row_groups = row_groups
        self.cfg = cfg
        self.reco_jet_pt_cut = reco_jet_pt_cut
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def build_tensors(self, data: ak.Array):
        jet_constituent_p4s = g.reinitialize_p4(data.reco_cand_p4s)
        gen_jet_tau_p4s = g.reinitialize_p4(data.gen_jet_tau_p4s)
        jet_p4s = g.reinitialize_p4(data.reco_jet_p4s)

        # ParticleTransformer features from https://arxiv.org/pdf/2202.03772, table 2
        cand_ParT_features = ak.Array(
            {
                "cand_deta": f.deltaEta(jet_constituent_p4s.eta, jet_p4s.eta),
                "cand_dphi": f.deltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
                "cand_logpt": np.log(jet_constituent_p4s.pt),
                "cand_loge": np.log(jet_constituent_p4s.energy),
                "cand_logptrel": np.log(jet_constituent_p4s.pt / jet_p4s.pt),
                "cand_logerel": np.log(jet_constituent_p4s.energy / jet_p4s.energy),
                "cand_deltaR": f.deltaR_etaPhi(
                    jet_constituent_p4s.eta,
                    jet_constituent_p4s.phi,
                    jet_p4s.eta,
                    jet_p4s.phi,
                ),
                "cand_charge": data.reco_cand_charge,
                "isElectron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 11, np.float32
                ),
                "isMuon": ak.values_astype(abs(data.reco_cand_pdg) == 13, np.float32),
                "isPhoton": ak.values_astype(abs(data.reco_cand_pdg) == 22, np.float32),
                "isChargedHadron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 211, np.float32
                ),
                "isNeutralHadron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 130, np.float32
                ),
            }
        )

        omni_features_wPID = ak.Array(
            {
                "part_pt": jet_constituent_p4s.pt,
                "part_mass": jet_constituent_p4s.mass,
                "part_etarel": f.deltaEta(jet_constituent_p4s.eta, jet_p4s.eta),
                "part_phirel": f.deltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
                "part_charge": data.reco_cand_charge,
                "part_isElectron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 11, np.float32
                ),
                "part_isMuon": ak.values_astype(
                    abs(data.reco_cand_pdg) == 13, np.float32
                ),
                "part_isPhoton": ak.values_astype(
                    abs(data.reco_cand_pdg) == 22, np.float32
                ),
                "part_isChargedHadron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 211, np.float32
                ),
                "part_isNeutralHadron": ak.values_astype(
                    abs(data.reco_cand_pdg) == 130, np.float32
                ),
            }
        )

        # raw particle kinematics, for LorentzNet and ParticleTransformer (attention matrix calculation)
        cand_kinematics = ak.Array(
            {
                "cand_px": jet_constituent_p4s.px,
                "cand_py": jet_constituent_p4s.py,
                "cand_pz": jet_constituent_p4s.pz,
                "cand_en": jet_constituent_p4s.energy,
            }
        )

        # additional track lifetime variables
        cand_lifetimes = ak.Array(
            {
                "cand_dz": data.reco_cand_dz,
                "cand_dz_err": data.reco_cand_dz_err,
                "cand_dxy": data.reco_cand_dxy,
                "cand_dxy_err": data.reco_cand_dxy_err,
            }
        )

        # Kinematic variables for the OmniJet
        cand_part_kinematics = ak.Array(
            {
                "part_pt": jet_constituent_p4s.pt,
                "part_etarel": jet_constituent_p4s.eta - jet_p4s.eta,
                "part_phirel": f.deltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
            }
        )

        cand_ParT_features_tensors = stack_and_pad_features(
            cand_ParT_features, self.cfg.max_cands
        )
        cand_kinematics_tensors = stack_and_pad_features(
            cand_kinematics, self.cfg.max_cands
        )
        cand_lifetimes_tensors = stack_and_pad_features(
            cand_lifetimes, self.cfg.max_cands
        )
        cand_omni_kinematics_tensors = stack_and_pad_features(
            cand_part_kinematics, self.cfg.max_cands
        )
        omni_features_wPID_tensors = stack_and_pad_features(
            omni_features_wPID, self.cfg.max_cands
        )

        cand_ParT_features_tensors = torch.tensor(
            cand_ParT_features_tensors, dtype=torch.float32
        )
        cand_kinematics_tensors = torch.tensor(
            cand_kinematics_tensors, dtype=torch.float32
        )
        cand_lifetimes_tensors = torch.tensor(
            cand_lifetimes_tensors, dtype=torch.float32
        )
        cand_omni_kinematics_tensors = torch.tensor(
            cand_omni_kinematics_tensors, dtype=torch.float32
        )
        omni_features_wPID_tensors = torch.tensor(
            omni_features_wPID_tensors, dtype=torch.float32
        )

        node_mask_tensors = torch.unsqueeze(
            torch.tensor(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            ak.ones_like(data.reco_cand_pdg),
                            self.cfg.max_cands,
                            clip=True,
                        ),
                        0,
                    )
                ),
                dtype=torch.bool,
            ),
            dim=1,
        )

        if not "weight" in data.fields:
            weight_tensors = torch.tensor(
                ak.ones_like(data.gen_jet_tau_decaymode), dtype=torch.float32
            )
        else:
            weight_tensors = torch.tensor(ak.to_numpy(data.weight), dtype=torch.float32)

        reco_jet_pt = torch.tensor(ak.to_numpy(jet_p4s.pt), dtype=torch.float32)
        gen_tau_pt = torch.tensor(ak.to_numpy(gen_jet_tau_p4s.pt), dtype=torch.float32)

        jet_regression_target = torch.log(gen_tau_pt / reco_jet_pt)
        gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
        reduced_gen_decay_modes = g.get_reduced_decaymodes(gen_jet_tau_decaymode)
        ohe_prepared_decay_modes = g.prepare_one_hot_encoding(reduced_gen_decay_modes)
        gen_jet_tau_decaymode_reduced = torch.tensor(ohe_prepared_decay_modes).long()

        gen_jet_tau_decaymode_exists = (
            torch.tensor(ak.to_numpy(data.gen_jet_tau_decaymode)) != -1
        ).long()

        # X, y, w
        return (
            # X - model inputs
            {
                "cand_kinematics": cand_kinematics_tensors,
                "cand_ParT_features": cand_ParT_features_tensors,
                "cand_lifetimes": cand_lifetimes_tensors,
                "cand_omni_kinematics": cand_omni_kinematics_tensors,
                "cand_omni_features_wPID": omni_features_wPID_tensors,
                "mask": node_mask_tensors,
                "reco_jet_pt": reco_jet_pt,
            },
            # y - targets
            {
                "reco_jet_pt": reco_jet_pt,
                "gen_tau_pt": gen_tau_pt,
                "jet_regression": jet_regression_target,
                "binary_classification": gen_jet_tau_decaymode_exists,
                "dm_multiclass": gen_jet_tau_decaymode_reduced,
            },
            # weights
            weight_tensors,
        )

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(
                math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            row_groups_start = worker_id * per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]

        for row_group in row_groups_to_process:
            # load one chunk from one file
            data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
            reco_jet_p4s = g.reinitialize_p4(data.reco_jet_p4s)
            data = data[reco_jet_p4s.pt >= self.reco_jet_pt_cut]
            tensors = self.build_tensors(data)

            # return individual jets from the dataset
            for ijet in range(len(data)):
                yield (
                    {k: v[ijet] for k, v in tensors[0].items()},
                    {k: v[ijet] for k, v in tensors[1].items()},
                    tensors[2][ijet],
                )
