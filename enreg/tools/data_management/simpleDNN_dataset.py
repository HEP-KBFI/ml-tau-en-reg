import json
import torch
import numpy as np
import awkward as ak
import enreg.tools.general as g
from omegaconf import OmegaConf
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from enreg.tools.models.SimpleDNN import DeepSet
import enreg.tools.data_management.features as f

class DeepSetDataset(Dataset):
    def __init__(self, data: ak.Array, cfg: DictConfig):

        self.cfg = cfg

        #jet p4        
        reco_jet_p4s = g.reinitialize_p4(data["reco_jet_p4s"])

        #per-jet PF candidates
        pf_p4s = g.reinitialize_p4(data["reco_cand_p4s"])

        #indices to map each PF candidate to jet 
        self.pf_lengths = ak.to_numpy(ak.num(pf_p4s))
        self.pf_startidx = np.cumsum(self.pf_lengths)
        self.pf_startidx -= self.pf_lengths

        #per PF candidate observables
        self.pf_px = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.px))), axis=-1).to(torch.float32)
        self.pf_py = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.py))), axis=-1).to(torch.float32)
        self.pf_pz = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.pz))), axis=-1).to(torch.float32)
        self.pf_pt = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.pt))), axis=-1).to(torch.float32)
        self.pf_eta = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.eta))), axis=-1).to(torch.float32)
        self.pf_phi = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.phi))), axis=-1).to(torch.float32)
        self.pf_energy = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.energy))), axis=-1).to(torch.float32)
        self.pf_pdg = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(np.abs(data["reco_cand_pdg"])))), axis=-1).to(torch.float32)
        self.pf_charge = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(data["reco_cand_charge"]))), axis=-1).to(torch.float32)
        ### 
        self.pf_logpt = torch.unsqueeze(torch.tensor(np.log(ak.to_numpy(ak.flatten(pf_p4s.pt)))), axis=-1).to(torch.float32)
        self.pf_loge = torch.unsqueeze(torch.tensor(np.log(ak.to_numpy(ak.flatten(pf_p4s.energy)))), axis=-1).to(torch.float32)

        self.pf_dphi = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaPhi(pf_p4s.phi, reco_jet_p4s.phi)))), axis=-1).to(torch.float32)
        self.pf_deta = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaEta(pf_p4s.eta, reco_jet_p4s.eta)))), axis=-1).to(torch.float32)

        self.pf_logptrel = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(1.0 - np.log(pf_p4s.pt/reco_jet_p4s.pt)))), axis=-1).to(torch.float32)
        self.pf_logerel = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(1.0 - np.log(pf_p4s.energy/reco_jet_p4s.energy)))), axis=-1).to(torch.float32)
        self.pf_deltaR = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaR_etaPhi(pf_p4s.eta, pf_p4s.phi, reco_jet_p4s.eta, reco_jet_p4s.phi)))), axis=-1).to(torch.float32)

        #per-jet observables
        self.reco_jet_pt = torch.unsqueeze(torch.tensor(ak.to_numpy(reco_jet_p4s.pt)), axis=-1).to(torch.float32)
        self.reco_jet_eta = torch.unsqueeze(torch.tensor(ak.to_numpy(reco_jet_p4s.eta)), axis=-1).to(torch.float32)
        self.reco_jet_mass = torch.unsqueeze(torch.tensor(ak.to_numpy(reco_jet_p4s.mass)), axis=-1).to(torch.float32)
        self.reco_jet_nptcl = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.num(pf_p4s.px))), axis=-1).to(torch.float32)

        #per-jet targets
        self.gen_jet_tau_p4s = g.reinitialize_p4(data["gen_jet_tau_p4s"])
        self.gen_jet_tau_decaymode = torch.tensor(ak.to_numpy(data["gen_jet_tau_decaymode"]), dtype=torch.long)
        self.gen_jet_tau_pt = torch.unsqueeze(torch.tensor(ak.to_numpy(self.gen_jet_tau_p4s.pt)), axis=-1).to(torch.float32)

        if not "weight" in data.fields:
            self.weight_tensors = torch.tensor(ak.ones_like(data.gen_jet_tau_decaymode), dtype=torch.float32)
        else:
            self.weight_tensors = torch.tensor(data.weight.to_list(), dtype=torch.float32)

    def __len__(self):
        return len(self.pf_lengths)

    def __getitem__(self, idx):
        assert(isinstance(idx, int))

        #get the indices of the PF candidates of the jet at 'idx'
        pf_range = range(self.pf_startidx[idx], self.pf_startidx[idx] + self.pf_lengths[idx])
        pf_deta = self.pf_deta[pf_range]
        pf_dphi = self.pf_dphi[pf_range]
        pf_logpt = self.pf_logpt[pf_range]
        pf_loge = self.pf_loge[pf_range]
        pf_logptrel = self.pf_logptrel[pf_range]
        pf_logerel = self.pf_logerel[pf_range]
        pf_deltaR = self.pf_deltaR[pf_range]
        pf_pdg = self.pf_pdg[pf_range]
        pf_charge = self.pf_charge[pf_range]

        pf_isele = pf_pdg == 11
        pf_ismu = pf_pdg == 13
        pf_isphoton = pf_pdg == 22
        pf_ischhad = pf_pdg == 211
        pf_isnhad = pf_pdg == 130

        # PF features (max_cands x 13)
        pfs = torch.concatenate([pf_deta, pf_dphi, pf_logptrel, pf_logpt, pf_logerel, pf_loge, pf_deltaR, pf_charge, pf_isele, pf_ismu, pf_isphoton, pf_ischhad, pf_isnhad], axis=-1)
        pfs_mask = torch.ones(pfs.shape[0], dtype=torch.float32)

        pfs[torch.isnan(pfs)] = 0
        pfs[torch.isinf(pfs)] = 0
        pfs = pfs[:self.cfg.max_cands]
        pfs = torch.nn.functional.pad(pfs, (0, 0, 0, self.cfg.max_cands - pfs.shape[0]))

        pfs_mask = pfs_mask[:self.cfg.max_cands]
        pfs_mask = torch.nn.functional.pad(pfs_mask, (0, self.cfg.max_cands - pfs_mask.shape[0]))

        # jet features (4)
        jets = torch.concatenate([self.reco_jet_pt[idx], self.reco_jet_eta[idx], self.reco_jet_mass[idx], self.reco_jet_nptcl[idx]], axis=-1)
        return (
            {
                "pfs": pfs,
                "pfs_mask": pfs_mask,
                "jets": jets,
            },
            {
                "reco_jet_pt": self.reco_jet_pt[idx],
                "gen_tau_pt": self.gen_jet_tau_pt[idx],
                "jet_regression": torch.log(self.gen_jet_tau_pt[idx]/self.reco_jet_pt[idx]).float(),
                "binary_classification": (self.gen_jet_tau_decaymode[idx] != -1).long(),
                "dm_multiclass": self.gen_jet_tau_decaymode[idx]
            },
            self.weight_tensors[idx],
        )



class DeepSetTauBuilder:
    def __init__(self, cfg: DictConfig, verbosity: int = 0):
        self.verbosity = verbosity
        self.is_energy_regression = cfg.builder.task == 'regression'
        self.is_dm_multiclass = cfg.builder.task == 'dm_multiclass'
        self.cfg = cfg

        if self.is_energy_regression:
            self.model = DeepSet(1)
            model_path = self.cfg.builder.regression.model_path
        elif self.is_dm_multiclass:
            self.model = DeepSet(16)
            model_path = self.cfg.builder.dm_multiclass.model_path
        else:
            model_path = self.cfg.builder.classification.model_path

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def print_config(self):
        primitive_cfg = OmegaConf.to_container(self.cfg)
        print(json.dumps(primitive_cfg, indent=4))

    def process_jets(self, data: ak.Array):
        print("::: Starting to process jets ::: ")
        dataset = TauDataset(data)

        jets = [dataset[i] for i in range(len(dataset))]
        jet_pfs = [jet.pfs for jet in jets]
        jet_pfs_mask = [jet.pfs_mask for jet in jets]
        jet_pfs_padded = torch.nn.utils.rnn.pad_sequence(jet_pfs, batch_first=True)
        jet_pfs_mask_padded = torch.nn.utils.rnn.pad_sequence(jet_pfs_mask, batch_first=True)
        pred = self.model(jet_pfs_padded, jet_pfs_mask_padded)

        if self.is_energy_regression:
            return {"tau_pt" : torch.exp(pred)[0] * dataset.reco_jet_pt}
        if self.is_dm_multiclass:
            return {"tau_dm": torch.argmax(pred, axis=-1)}
