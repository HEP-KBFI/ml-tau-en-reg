import math
import vector
import awkward as ak
from torch.utils.data import Dataset
from enreg.tools.models.LorentzNet import psi, LorentzNet
from sklearn.preprocessing import OneHotEncoder
from omegaconf import DictConfig
from omegaconf import OmegaConf
from enreg.tools import general as g
import numpy as np
import json
import torch
from torch import nn


class LorentzNetDataset(Dataset):
    def __init__(self, data: ak.Array, cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.preselection()
        self.num_jets = len(self.data.reco_jet_p4s)
        self.build_tensors()

    def preselection(self):
        """Chooses jets based on preselection criteria specified in the configuration"""
        jet_constituent_p4s = g.reinitialize_p4(self.data.reco_cand_p4s)
        jet_p4s = g.reinitialize_p4(self.data.reco_jet_p4s)
        no_mask = ak.ones_like(self.data.gen_jet_tau_decaymode, dtype=bool)
        min_jet_theta_mask = no_mask if self.cfg.min_jet_theta == -1 else jet_p4s.theta >= self.cfg.min_jet_theta
        max_jet_theta_mask = no_mask if self.cfg.max_jet_theta == -1 else jet_p4s.theta <= self.cfg.max_jet_theta
        min_jet_pt_mask = no_mask if self.cfg.min_jet_pt == -1 else jet_p4s.pt >= self.cfg.min_jet_pt
        max_jet_pt_mask = no_mask if self.cfg.max_jet_pt == -1 else jet_p4s.pt <= self.cfg.max_jet_pt
        preselection_mask = min_jet_theta_mask * max_jet_theta_mask * min_jet_pt_mask * max_jet_pt_mask
        self.data = ak.Array({field: self.data[field] for field in self.data.fields})[preselection_mask]

    def build_tensors(self):
        jet_constituent_p4s = g.reinitialize_p4(self.data.reco_cand_p4s)
        self.gen_jet_tau_p4s = g.reinitialize_p4(self.data.gen_jet_tau_p4s)
        self.jet_p4s = g.reinitialize_p4(self.data.reco_jet_p4s)
        cand_kinematics = ak.Array({
            "cand_px": jet_constituent_p4s.px,
            "cand_py": jet_constituent_p4s.py,
            "cand_pz": jet_constituent_p4s.pz,
            "cand_en": jet_constituent_p4s.energy,
        })
        for cand_property in cand_kinematics.fields:
            cand_kinematics[cand_property] = ak.fill_none(
                ak.pad_none(cand_kinematics[cand_property], self.cfg.max_cands, clip=True),
                0,
            )
        x_tensor_full = np.swapaxes(
            np.swapaxes(np.array([cand_kinematics[feature].to_list() for feature in cand_kinematics.fields]), 0, 1), 1, 2)
        self.x_tensors = torch.tensor(x_tensor_full, dtype=torch.float32)
        self.x_is_one_hot_encoded = torch.tensor(
            [[self.cfg.one_hot_encoding[feature] for feature in cand_kinematics.fields]] * len(self.jet_p4s),
            dtype=torch.bool
        )
        if self.cfg.use_pdgId:
            cand_features = ak.Array({
                "isElectron" : ak.values_astype(abs(self.data.reco_cand_pdg) == 11, "float32"),
                "isMuon" : ak.values_astype(abs(self.data.reco_cand_pdg) == 13, "float32"),
                "isPhoton" : ak.values_astype(abs(self.data.reco_cand_pdg) == 22, "float32"),
                "isNeutralHadron" : ak.values_astype(abs(self.data.reco_cand_pdg) == 130, "float32"),
                "isChargedHadron" : ak.values_astype(abs(self.data.reco_cand_pdg) == 211, "float32"),
                "isProton": ak.zeros_like(self.data.reco_cand_pdg, dtype="float32"),
                "cand_charge": self.data.reco_cand_charge,
            })
        else:
            cand_features = ak.Array({
                "mass_psi": psi(torch.tensor(ak.fill_none(
                    ak.pad_none(jet_constituent_p4s.mass, self.cfg.max_cands, clip=True), 0,), dtype=torch.float32)),
                "pad": ak.zeros_like(self.data.reco_cand_charge),  # as in CV implementation. Reason ?
            })
        for cand_feature in cand_features.fields:
            cand_features[cand_feature] = ak.fill_none(
                ak.pad_none(cand_features[cand_feature], self.cfg.max_cands, clip=True),
                0,
            )
        self.scalars_tensors = np.swapaxes(
            np.swapaxes(
                np.array([cand_features[feature].to_list() for feature in cand_features.fields]),
                0, 1
            ), 1, 2
        )
        self.scalars_tensors = torch.tensor(self.scalars_tensors, dtype=torch.float32)
        self.scalars_is_one_hot_encoded = torch.tensor(
            [[self.cfg.one_hot_encoding[feature] for feature in cand_features.fields]] * len(self.jet_p4s),
            dtype=torch.bool
        )
        self.node_mask_tensors = torch.tensor(
                ak.fill_none(ak.pad_none(ak.ones_like(self.data.reco_cand_pdg), self.cfg.max_cands, clip=True), 0,),
                dtype=torch.float32
        )
        if self.cfg.beams.add:
            beam1_p4 = [math.sqrt(1 + self.cfg.beams.mass**2), 0.0, 0.0, +1.0]
            beam2_p4 = [math.sqrt(1 + self.cfg.beams.mass**2), 0.0, 0.0, -1.0]
            x_beams = torch.tensor([[beam1_p4, beam2_p4]] * len(self.jet_p4s), dtype=torch.float32)
            self.x_tensors = torch.cat([x_beams, self.x_tensors], dim=1)
            if self.cfg.use_pdgId:
                beam_features = ak.Array({
                    "isElectron" : np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "isMuon" : np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "isPhoton" : np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "isNeutralHadron" : np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "isChargedHadron" : np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "isProton": np.zeros((len(self.jet_p4s), 2), dtype=np.float32),
                    "cand_charge": ak.Array([[+1.0, -1.0]] * len(self.jet_p4s))
                })
            else:
                beam_features = ak.Array({
                    "mass_psi": psi(torch.tensor(ak.Array([[beam_mass, beam_mass]] * len(self.jet_p4s)), dtype=torch.float32)),
                    "pad": ak.Array([[0.0, 0.0]] * len(self.jet_p4s))
                })
            scalars_beams = np.swapaxes(
                np.swapaxes(
                    np.array([beam_features[feature].to_list() for feature in beam_features.fields]),
                    0, 1
                ), 1, 2
            )
            scalars_beams = torch.tensor(scalars_beams)
            self.scalars_tensors = torch.cat([scalars_beams, self.scalars_tensors], dim=1).float()

            node_mask_beams = torch.tensor(np.ones((len(self.jet_p4s), 2)), dtype=torch.float32)
            self.node_mask_tensors = torch.unsqueeze(
                torch.cat([node_mask_beams, self.node_mask_tensors], dim=1),
                dim=2
            )

        if not "weight" in self.data.fields:
            self.weight_tensors = torch.tensor(ak.ones_like(self.data.gen_jet_tau_decaymode), dtype=torch.float32)
        else:
            self.weight_tensors = torch.tensor(self.data.weight.to_list(), dtype=torch.float32)
            
        self.reco_jet_pt = torch.tensor(self.jet_p4s.pt, dtype=torch.float32)
        self.reco_jet_energy = torch.tensor(self.jet_p4s.energy, dtype=torch.float32)

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return (
                {
                    "x": self.x_tensors[idx],
                    "x_is_one_hot_encoded": self.x_is_one_hot_encoded,
                    "scalars": self.scalars_tensors[idx],
                    "scalars_is_one_hot_encoded": self.scalars_is_one_hot_encoded,
                    "mask": self.node_mask_tensors[idx],
                    "reco_jet_pt": self.reco_jet_pt[idx],
                },
                {
                    "reco_jet_pt": self.reco_jet_pt[idx],
                    "gen_tau_pt": self.gen_jet_tau_p4s[idx].pt,
                    "jet_regression": torch.log(self.gen_jet_tau_p4s[idx].pt/self.reco_jet_pt[idx]).to(torch.float),
                    "binary_classification": torch.tensor(self.data.gen_jet_tau_decaymode[idx] != -1, dtype=torch.long),
                    "dm_multiclass": torch.tensor(self.data.gen_jet_tau_decaymode[idx], dtype=torch.long)
                },
                self.weight_tensors[idx],
            )
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))


class LorentzNetTauBuilder:
    def __init__(self, cfg: DictConfig, verbosity: int = 0):
        self.verbosity = verbosity
        if cfg.builder.task == 'regression':
            self.kind = "is_energy_regression"
        elif cfg.builder.task == 'dm_multiclass':
            self.kind = "is_dm_multiclass"
        self.cfg = cfg
        if self.cfg.feature_standardization.standardize_inputs:
            self.transform = f.FeatureStandardization(
                method=self.cfg.feature_standardization.method,
                features=["x", "scalars"],
                feature_dim=1,
                verbosity=self.verbosity,
            )
            self.transform.load_params(self.cfg.feature_standardization.path)
        self.n_scalar = 7 if cfg.dataset.use_pdgId else 2
        if self.kind == "is_energy_regression":
            self.num_classes = 1
        elif self.kind == "is_dm_multiclass":
            self.num_classes = 16
        else:
            self.num_classes = 2
        self.model = LorentzNet(
            n_scalar=self.n_scalar,
            n_hidden=cfg.training.n_hidden,
            n_class=self.num_classes,
            n_layers=cfg.training.n_layers,
            c_weight=cfg.training.c_weight,
            dropout=cfg.training.dropout,
            verbosity=self.verbosity
        )
        if self.kind == "is_energy_regression":
            model_path = self.cfg.builder.regression.model_path
        if self.kind == "is_dm_multiclass":
            model_path = self.cfg.builder.dm_multiclass.model_path
        else:
            self.cfg.builder.classification.model_path
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def print_config(self):
        primitive_cfg = OmegaConf.to_container(self.cfg)
        print(json.dumps(primitive_cfg, indent=4))

    def process_jets(self, data: ak.Array):
        print("::: Starting to process jets ::: ")
        dataset = LorentzNetDataset(data, self.cfg.dataset, self.kind)
        if self.cfg.feature_standardization.standardize_inputs:
            X = {
                "x": dataset.x_tensors,
                "scalars": dataset.scalars_tensors,
                "mask": dataset.node_mask_tensors,
            }
            X_transformed = self.transform(X)
            x_tensor = X_transformed["x"]
            scalars_tensor = X_transformed["scalars"]
            node_mask_tensor = X_transformed["mask"]
        else:
            x_tensor = dataset.x_tensors
            scalars_tensors = dataset.scalars_tensors
            node_mask_tensor = dataset.node_mask_tensors
        with torch.no_grad():
            pred = self.model(x_tensor, scalars_tensors, node_mask_tensor)
        if self.kind == "is_energy_regression":
            return {"tau_pt" : torch.exp(pred)[0] * dataset.reco_jet_pt}
        elif self.kind == "is_dm_multiclass":
            return {"tau_dm": torch.argmax(pred, axis=-1)}
        else:
            pred = torch.softmax(pred, dim=1)
            tauClassifier = pred[:, 1]  # pred_mask_tensor not needed as the tensors from dataset contain only preselected ones
            tauClassifier = list(tauClassifier.detach().numpy())
            tau_p4s = g.reinitialize_p4(data["reco_jet_p4s"])
            tauSigCand_p4s = data["reco_cand_p4s"]
            tauCharges = np.zeros(len(dataset))
            tau_decaymode = np.zeros(len(dataset))
            return {
                "tau_p4s": tau_p4s,
                "tauSigCand_p4s": tauSigCand_p4s,
                "tauClassifier": tauClassifier,
                "tau_charge": tauCharges,
                "tau_decaymode": tau_decaymode,
            }