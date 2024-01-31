import torch
import json
import numpy as np
import awkward as ak
import enreg.tools.general as g
from omegaconf import OmegaConf
from omegaconf import DictConfig
from torch.utils.data import Dataset
import enreg.tools.data_management.features as f
from enreg.tools.models.ParticleTransformer import ParticleTransformer


class ParticleTransformerDataset(Dataset):
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
        jet_p4s = g.reinitialize_p4(self.data.reco_jet_p4s)
        cand_features = ak.Array({
            "cand_deta": f.deltaEta(jet_constituent_p4s.eta, jet_p4s.eta),  # Could also use dTheta
            "cand_dphi": f.deltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
            "cand_logpt": np.log(jet_constituent_p4s.pt),
            "cand_loge": np.log(jet_constituent_p4s.energy),
            "cand_logptrel": np.log(jet_constituent_p4s.pt / jet_p4s.pt),
            "cand_logerel": np.log(jet_constituent_p4s.energy / jet_p4s.energy),
            "cand_deltaR": f.deltaR_etaPhi(
                jet_constituent_p4s.eta, jet_constituent_p4s.phi, jet_p4s.eta, jet_p4s.phi), #angle3d
        })
        if self.cfg.use_pdgId:
            cand_features["cand_charge"] = self.data.reco_cand_charge
            cand_features["isElectron"] = ak.values_astype(abs(self.data.reco_cand_pdg) == 11, "float64")
            cand_features["isMuon"] = ak.values_astype(abs(self.data.reco_cand_pdg) == 13, "float64")
            cand_features["isPhoton"] = ak.values_astype(abs(self.data.reco_cand_pdg) == 22, "float64")
            cand_features["isChargedHadron"] = ak.values_astype(abs(self.data.reco_cand_pdg) == 211, "float64")
            cand_features["isNeutralHadron"] = ak.values_astype(abs(self.data.reco_cand_pdg) == 130, "float64")
        if self.cfg.use_lifetime:
            # There is some problem with the lifetime variables as they do no exactly match the ones on CV implementation
            charge_mask = ak.values_astype(np.abs(self.data.reco_cand_charge) == 1, "int64")
            d0_mask = ak.values_astype(np.abs(self.data.reco_cand_d0) > -99, "int64")
            dz_mask = ak.values_astype(np.abs(self.data.reco_cand_dz) > -99, "int64")
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
        for cand_feature in cand_features.fields:
            cand_features[cand_feature] = ak.fill_none(
                ak.pad_none(cand_features[cand_feature], self.cfg.max_cands, clip=True),
                0,
            )
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
        # Swpping the axes such that it has the shape of (nJets, nFeatures, nParticles)
        x_tensor_full = np.swapaxes(np.array([cand_features[feature].to_list() for feature in cand_features.fields]), 0, 1)
        v_tensor_full = np.swapaxes(
                np.array([cand_kinematics[feature].to_list() for feature in cand_kinematics.fields]), 0, 1)
        # Splitting the full_tensor nJets subtensors:
        self.x_tensors = torch.tensor(x_tensor_full, dtype=torch.float32)
        self.v_tensors = torch.tensor(v_tensor_full, dtype=torch.float32)
        self.node_mask_tensors = torch.unsqueeze(
            torch.tensor(
                ak.fill_none(ak.pad_none(ak.ones_like(self.data.reco_cand_pdg), self.cfg.max_cands, clip=True), 0,),
                dtype=torch.float32
            ),
            dim=1
        )
        self.x_is_one_hot_encoded = torch.tensor(
            [[self.cfg.one_hot_encoding[feature] for feature in cand_features.fields]] * len(jet_p4s),
            dtype=torch.bool
        )
        self.v_is_one_hot_encoded = torch.tensor(
            [[self.cfg.one_hot_encoding[feature] for feature in cand_kinematics.fields]] * len(jet_p4s),
            dtype=torch.bool
        )
        if not "weight" in self.data.fields:
            self.weight_tensors = torch.tensor(ak.ones_like(self.data.gen_jet_tau_decaymode), dtype=torch.float32)
        else:
            self.weight_tensors = torch.tensor(self.data.weight.to_list(), dtype=torch.float32)
        self.y_tensors = torch.tensor(self.data.gen_jet_tau_decaymode != -1, dtype=int)

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return (
                {
                    "v": self.v_tensors[idx],
                    "v_is_one_hot_encoded": self.v_is_one_hot_encoded,
                    "x": self.x_tensors[idx],
                    "x_is_one_hot_encoded": self.x_is_one_hot_encoded,
                    "mask": self.node_mask_tensors[idx],
                },
                self.y_tensors[idx],
                self.weight_tensors[idx],
            )
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))


class ParticleTransformerTauBuilder:
    def __init__(self, cfg: DictConfig, verbosity: int = 0):
        print("::: ParticleTransformer :::")
        self.verbosity = verbosity
        self.cfg = cfg
        if self.cfg.builder.feature_standardization.standardize_inputs:
            self.transform = f.FeatureStandardization(
                method=self.cfg.builder.feature_standardization.method,
                features=["x", "v"],
                feature_dim=1,
                verbosity=self.verbosity,
            )
            self.transform.load_params(self.cfg.builder.feature_standardization.path)

        input_dim = 7
        if self.cfg.dataset.use_pdgId:
            input_dim += 6
        if self.cfg.dataset.use_lifetime:
            input_dim += 4
        #  TODO: check this out if needs a change?
        self.model = ParticleTransformer(
            input_dim=input_dim,
            num_classes=2,
            use_pre_activation_pair=False,
            for_inference=False,  # CV: keep same as for training and apply softmax function on NN output manually
            use_amp=False,
            metric=cfg.builder.metric,
            verbosity=verbosity,
        )
        self.model.load_state_dict(torch.load(self.cfg.builder.model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def print_config(self):
        primitive_cfg = OmegaConf.to_container(self.cfg)
        print(json.dumps(primitive_cfg, indent=4))

    def process_jets(self, data: ak.Array):
        print("::: Starting to process jets ::: ")
        dataset = ParticleTransformerDataset(data, self.cfg.dataset)
        if self.cfg.builder.feature_standardization.standardize_inputs:
            X = {
                "x": dataset.x_tensors,
                "v": dataset.v_tensors,
                "mask": dataset.node_mask_tensors,
            }
            X_transformed = self.transform(X)
            x_tensor = X_transformed["x"]
            v_tensor = X_transformed["v"]
            node_mask_tensor = X_transformed["mask"]
        else:
            x_tensor = dataset.x_tensors
            v_tensor = dataset.v_tensors
            node_mask_tensor = dataset.node_mask_tensors
        pred = self.model(x_tensor, v_tensor, node_mask_tensor)
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
