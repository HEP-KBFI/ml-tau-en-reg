import torch
import json
import math
import numpy as np
import awkward as ak
import enreg.tools.general as g
from omegaconf import OmegaConf
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch import nn
import enreg.tools.data_management.features as f
from enreg.tools.models.ParticleTransformer import ParticleTransformer

def stack_and_pad_features(cand_features, max_cands):
    cand_features_tensors = np.stack([ak.pad_none(cand_features[feat], max_cands, clip=True) for feat in cand_features.fields], axis=-1)
    cand_features_tensors = ak.to_numpy(ak.fill_none(cand_features_tensors, 0))
    # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
    cand_features_tensors = np.swapaxes(cand_features_tensors, 1, 2)

    cand_features_tensors[np.isnan(cand_features_tensors)] = 0
    cand_features_tensors[np.isinf(cand_features_tensors)] = 0
    return cand_features_tensors

class ParticleTransformerDataset(Dataset):
    def __init__(self, data: ak.Array, cfg: DictConfig, do_preselection=True):
        self.data = data
        self.cfg = cfg
        if do_preselection:
            self.preselection()
        self.num_jets = len(self.data.reco_jet_p4s)
        self.build_tensors()

    def preselection(self):
        print("ParticleTransformer.preselection")
        """Chooses jets based on preselection criteria specified in the configuration"""
        jet_constituent_p4s = g.reinitialize_p4(self.data.reco_cand_p4s)
        jet_p4s = g.reinitialize_p4(self.data.reco_jet_p4s)
        no_mask = ak.ones_like(self.data.gen_jet_tau_decaymode, dtype=bool)
        min_jet_theta_mask = no_mask if self.cfg.min_jet_theta == -1 else jet_p4s.theta >= self.cfg.min_jet_theta
        max_jet_theta_mask = no_mask if self.cfg.max_jet_theta == -1 else jet_p4s.theta <= self.cfg.max_jet_theta
        min_jet_pt_mask = no_mask if self.cfg.min_jet_pt == -1 else jet_p4s.pt >= self.cfg.min_jet_pt
        max_jet_pt_mask = no_mask if self.cfg.max_jet_pt == -1 else jet_p4s.pt <= self.cfg.max_jet_pt
        preselection_mask = min_jet_theta_mask * max_jet_theta_mask * min_jet_pt_mask * max_jet_pt_mask
        print("mask: {}/{}".format(np.sum(preselection_mask), len(preselection_mask)))
        self.data = ak.Array({field: self.data[field] for field in self.data.fields})[preselection_mask]

    def build_tensors(self):
        print("ParticleTransformer.build_tensors")
        jet_constituent_p4s = g.reinitialize_p4(self.data.reco_cand_p4s)
        self.gen_jet_tau_p4s = g.reinitialize_p4(self.data.gen_jet_tau_p4s)
        self.jet_p4s = g.reinitialize_p4(self.data.reco_jet_p4s)

        cand_features = ak.Array({
            "cand_charge": self.data.reco_cand_charge,
            "isElectron": ak.values_astype(abs(self.data.reco_cand_pdg) == 11, np.float32),
            "isMuon": ak.values_astype(abs(self.data.reco_cand_pdg) == 13, np.float32),
            "isPhoton": ak.values_astype(abs(self.data.reco_cand_pdg) == 22, np.float32),
            "isChargedHadron": ak.values_astype(abs(self.data.reco_cand_pdg) == 211, np.float32),
            "isNeutralHadron": ak.values_astype(abs(self.data.reco_cand_pdg) == 130, np.float32),
            "cand_deta": f.deltaEta(jet_constituent_p4s.eta, self.jet_p4s.eta),  # Could also use dTheta
            "cand_dphi": f.deltaPhi(jet_constituent_p4s.phi, self.jet_p4s.phi),
            "cand_logpt": np.log(jet_constituent_p4s.pt),
            "cand_loge": np.log(jet_constituent_p4s.energy),
            "cand_logptrel": np.log(jet_constituent_p4s.pt / self.jet_p4s.pt),
            "cand_logerel": np.log(jet_constituent_p4s.energy / self.jet_p4s.energy),
            "cand_deltaR": f.deltaR_etaPhi(
                jet_constituent_p4s.eta, jet_constituent_p4s.phi, self.jet_p4s.eta, self.jet_p4s.phi), #angle3d
        })

        if self.cfg.use_lifetime:
            print("use_lifetime")
            # There is some problem with the lifetime variables as they do no exactly match the ones on CV implementation
            charge_mask = ak.values_astype(np.abs(self.data.reco_cand_charge) == 1, np.int64)
            d0_mask = ak.values_astype(np.abs(self.data.reco_cand_d0) > -99, np.int64)
            dz_mask = ak.values_astype(np.abs(self.data.reco_cand_dz) > -99, np.int64)
            total_mask = charge_mask * d0_mask * dz_mask
            cand_features["cand_d0"] = (np.tanh(self.data.reco_cand_d0)) * total_mask
            cand_features["cand_d0_err"] = (
                np.tanh(self.data.reco_cand_d0)
                / ak.max(
                    ak.Array([0.01 * self.data.reco_cand_d0, self.data.reco_cand_d0_err]),
                    axis=0,
                )
            ) * total_mask
            cand_features["cand_dz"] = (np.tanh(self.data.reco_cand_dz)) * total_mask
            cand_features["cand_dz_err"] = (
                np.tanh(self.data.reco_cand_dz)
                / ak.max(
                    ak.Array([0.01 * self.data.reco_cand_dz, self.data.reco_cand_dz_err]),
                    axis=0,
                )
            ) * total_mask

        cand_kinematics = ak.Array({
            "cand_px": jet_constituent_p4s.px,
            "cand_py": jet_constituent_p4s.py,
            "cand_pz": jet_constituent_p4s.pz,
            "cand_en": jet_constituent_p4s.energy,
        })

        print("creating padded tensors")
        cand_features_tensors = stack_and_pad_features(cand_features, self.cfg.max_cands)
        cand_kinematics_tensors = stack_and_pad_features(cand_kinematics, self.cfg.max_cands)

        self.cand_features_tensors = torch.tensor(cand_features_tensors, dtype=torch.float32)
        self.cand_kinematics_tensors = torch.tensor(cand_kinematics_tensors, dtype=torch.float32)
        print("cand_features_tensors={} cand_kinematics_tensors={}".format(self.cand_features_tensors.shape, self.cand_kinematics_tensors.shape))

        #for LorentzNet, add two additional fake particles (i.e. beams)
        #these are the Lorentz-invariant quantities and will be stacked with cand_kinematics in the network
        beam1_p4 = [math.sqrt(1 + self.cfg.beams.mass**2), 0.0, 0.0, +1.0]
        beam2_p4 = [math.sqrt(1 + self.cfg.beams.mass**2), 0.0, 0.0, -1.0]
        self.beams_kinematics_tensor = torch.tensor([[beam1_p4, beam2_p4]] * len(self.jet_p4s), dtype=torch.float32)
        self.beams_kinematics_tensor = torch.swapaxes(self.beams_kinematics_tensor, 1, 2)

        #these are the various non-Lorentz-invariant quantities
        #these will be stacked with cand_features in the network
        beams_features = ak.Array({
            "cand_charge": ak.Array([[+1.0, -1.0]] * len(self.jet_p4s)),
        })
        for field in cand_features.fields:
            if field == "cand_charge":
                continue
            beams_features[field] = ak.Array([[0.0, 0.0]] * len(self.jet_p4s))

        beams_features = stack_and_pad_features(beams_features, 2)
        self.beams_features_tensor = torch.tensor(beams_features).float()

        self.beams_mask_tensor = torch.tensor(np.ones((len(self.jet_p4s), 1, 2)), dtype=torch.float32)

        print("creating mask")
        self.node_mask_tensors = torch.unsqueeze(
            torch.tensor(
                ak.to_numpy(ak.fill_none(ak.pad_none(ak.ones_like(self.data.reco_cand_pdg), self.cfg.max_cands, clip=True), 0,)),
                dtype=torch.float32
            ),
            dim=1
        )
        print("node_mask_tensors={}".format(self.node_mask_tensors.shape))

        print("creating weights")
        if not "weight" in self.data.fields:
            self.weight_tensors = torch.tensor(ak.ones_like(self.data.gen_jet_tau_decaymode), dtype=torch.float32)
        else:
            self.weight_tensors = torch.tensor(ak.to_numpy(self.data.weight), dtype=torch.float32)
        print("weight_tensors={}".format(self.weight_tensors))
        print("weight_tensors values: {}".format(self.weight_tensors[:10]))

        self.reco_jet_pt = torch.tensor(ak.to_numpy(self.jet_p4s.pt), dtype=torch.float32)
        self.reco_jet_energy = torch.tensor(ak.to_numpy(self.jet_p4s.energy), dtype=torch.float32)

        self.jet_regression_target = torch.log(torch.tensor(ak.to_numpy(self.gen_jet_tau_p4s.pt/self.reco_jet_pt), dtype=torch.float32))
        self.gen_jet_tau_decaymode = torch.tensor(ak.to_numpy(self.data.gen_jet_tau_decaymode)).long()
        self.gen_jet_tau_decaymode_exists = (self.gen_jet_tau_decaymode != -1).long()
        print("ParticleTransformer.build_tensors done")

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return (
                {
                    "cand_kinematics": self.cand_kinematics_tensors[idx],
                    "cand_features": self.cand_features_tensors[idx],

                    #for LorentzNet
                    "beam_kinematics": self.beams_kinematics_tensor[idx],
                    "beam_features": self.beams_features_tensor[idx],
                    "beam_mask": self.beams_mask_tensor[idx],

                    "mask": self.node_mask_tensors[idx],

                    "reco_jet_pt": self.reco_jet_pt[idx],
                },
                {
                    "reco_jet_pt": self.reco_jet_pt[idx],
                    "gen_tau_pt": self.gen_jet_tau_p4s[idx].pt,
                    "jet_regression": self.jet_regression_target[idx],
                    "binary_classification": self.gen_jet_tau_decaymode_exists[idx],
                    "dm_multiclass": self.gen_jet_tau_decaymode[idx]
                },
                self.weight_tensors[idx],
            )
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))