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


#given multiple jets with a variable number of PF candidates per jet, create 3d-padded arrays
#in the shape [Njets, Npfs_max, Nfeat]
def pad_collate(jets):
    pfs = [jet.pfs for jet in jets]
    pfs_mask = [jet.pfs_mask for jet in jets]
    gen_tau_label = [jet.gen_tau_label for jet in jets]
    gen_tau_pt = [jet.gen_tau_pt for jet in jets]
    reco_jet_pt = [jet.reco_jet_pt for jet in jets]
    pfs = torch.nn.utils.rnn.pad_sequence(pfs, batch_first=True)
    pfs_mask = torch.nn.utils.rnn.pad_sequence(pfs_mask, batch_first=True)
    gen_tau_label = torch.concatenate(gen_tau_label, axis=0)
    gen_tau_pt = torch.concatenate(gen_tau_pt, axis=0)
    reco_jet_pt = torch.concatenate(reco_jet_pt, axis=0)
    return Jet(
        pfs=pfs,
        pfs_mask=pfs_mask,
        reco_jet_pt=reco_jet_pt,
        gen_tau_label=gen_tau_label,
        gen_tau_pt=gen_tau_pt
    )


class Jet:
    def __init__(
        self,
        pfs: torch.Tensor,
        pfs_mask: torch.Tensor,
        reco_jet_pt: torch.Tensor,
        gen_tau_label: torch.Tensor,
        gen_tau_pt: torch.Tensor
    ):
        self.pfs = pfs
        self.pfs_mask = pfs_mask
        self.reco_jet_pt = reco_jet_pt
        self.gen_tau_label = gen_tau_label
        self.gen_tau_pt = gen_tau_pt


class TauDataset(Dataset):
    def __init__(self, data: ak.Array):
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
        reco_jet_p4s = g.reinitialize_p4(data["reco_jet_p4s"])

        self.pf_dphi = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaPhi(pf_p4s.phi, reco_jet_p4s.phi)))), axis=-1).to(torch.float32)
        self.pf_deta = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaEta(pf_p4s.eta, reco_jet_p4s.eta)))), axis=-1).to(torch.float32)

        self.pf_logptrel = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(np.log(pf_p4s.pt/reco_jet_p4s.pt)))), axis=-1).to(torch.float32)
        self.pf_logerel = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(np.log(pf_p4s.energy/reco_jet_p4s.energy)))), axis=-1).to(torch.float32)
        self.pf_deltaR = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(f.deltaR_etaPhi(pf_p4s.eta, pf_p4s.phi, reco_jet_p4s.eta, reco_jet_p4s.phi)))), axis=-1).to(torch.float32)

        #per-jet observables
        self.reco_jet_pts = torch.unsqueeze(torch.tensor(ak.to_numpy(reco_jet_p4s.pt)), axis=-1).to(torch.float32)
        #per-jet targets
        gen_jet_p4s = g.reinitialize_p4(data["gen_jet_tau_p4s"])
        self.gen_tau_labels = torch.unsqueeze(torch.tensor(ak.to_numpy(data["gen_jet_tau_decaymode"])), axis=-1).to(torch.float32)
        self.gen_tau_pts = torch.unsqueeze(torch.tensor(ak.to_numpy(gen_jet_p4s.pt)), axis=-1).to(torch.float32)

    def __len__(self):
        return len(self.pf_lengths)

    def __getitem__(self, idx):
        assert(isinstance(idx, int))

        #get the indices of the PF candidates of the jet at 'idx'
        pf_range = range(self.pf_startidx[idx], self.pf_startidx[idx] + self.pf_lengths[idx])
        pf_px = self.pf_px[pf_range]
        pf_py = self.pf_py[pf_range]
        pf_pz = self.pf_pz[pf_range]
        pf_pt = self.pf_pt[pf_range]
        pf_eta = self.pf_eta[pf_range]
        pf_phi = self.pf_phi[pf_range]
        pf_energy = self.pf_energy[pf_range]
        pf_logpt = self.pf_logpt[pf_range]
        pf_loge = self.pf_loge[pf_range]
        pf_dphi = self.pf_dphi[pf_range]
        pf_deta = self.pf_deta[pf_range]
        pf_logptrel = self.pf_logptrel[pf_range]
        pf_logerel = self.pf_logerel[pf_range]
        pf_deltaR = self.pf_deltaR[pf_range]
        pf_pdg = self.pf_pdg[pf_range] #FIXME: this could be better as a one-hot encoded value, rather than a floating-point PDGID value
        pf_charge = self.pf_charge[pf_range]
        # pfs = torch.concatenate([pf_px, pf_py, pf_pz, pf_energy, pf_pdg, pf_charge], axis=-1)
        pfs = torch.concatenate([pf_pt, pf_eta, torch.sin(pf_phi), torch.cos(pf_phi), pf_energy, pf_pdg, pf_charge], axis=-1)
        return Jet(
            pfs=pfs,
            pfs_mask=torch.ones(pfs.shape[0], dtype=torch.float32),
            reco_jet_pt=self.reco_jet_pts[idx],
            gen_tau_label=self.gen_tau_labels[idx],
            gen_tau_pt=self.gen_tau_pts[idx]
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
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=500,
        #     num_workers=6,
        #     prefetch_factor=20,
        #     shuffle=False,
        #     collate_fn=pad_collate
        # )
        # # dataloader = dataloader[0]
        # for ibatch, batched_jets in enumerate(dataloader):
        #     pfs = batched_jets.pfs
        #     pfs_mask = batched_jets.pfs_mask
        #     reco_jet_pt = batched_jets.reco_jet_pt
        #     gen_tau_label = batched_jets.gen_tau_label
        #     with torch.no_grad():
        #         pred = self.model(pfs, pfs_mask)
        # return {"tau_dm" : torch.exp(pred)[0]}
        if self.is_energy_regression:
            return {"tau_pt" : torch.exp(pred)[0] * dataset.reco_jet_pt}
        if self.is_dm_multiclass:
            return {"tau_dm": torch.argmax(pred, axis=-1)}
