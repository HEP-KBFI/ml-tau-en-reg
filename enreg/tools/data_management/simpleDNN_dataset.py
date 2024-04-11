import torch
import numpy as np
import awkward as ak
import enreg.tools.general as g
from omegaconf import DictConfig
from torch.utils.data import Dataset


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
        self.pf_pt = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.pt))), axis=-1).to(torch.float32)
        self.pf_eta = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.eta))), axis=-1).to(torch.float32)
        self.pf_phi = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.phi))), axis=-1).to(torch.float32)
        self.pf_energy = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(pf_p4s.energy))), axis=-1).to(torch.float32)
        self.pf_pdg = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(np.abs(data["reco_cand_pdg"])))), axis=-1).to(torch.float32)
        self.pf_charge = torch.unsqueeze(torch.tensor(ak.to_numpy(ak.flatten(data["reco_cand_charge"]))), axis=-1).to(torch.float32)
        #per-jet observables
        reco_jet_p4s = g.reinitialize_p4(data["reco_jet_p4s"])
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
        pf_pt = self.pf_pt[pf_range]
        pf_eta = self.pf_eta[pf_range]
        pf_phi = self.pf_phi[pf_range]
        pf_energy = self.pf_energy[pf_range]
        pf_pdg = self.pf_pdg[pf_range] #FIXME: this could be better as a one-hot encoded value, rather than a floating-point PDGID value
        pf_charge = self.pf_charge[pf_range]
        pfs = torch.concatenate([pf_pt, pf_eta, torch.sin(pf_phi), torch.cos(pf_phi), pf_energy, pf_pdg, pf_charge], axis=-1)
        return Jet(
            pfs=pfs,
            pfs_mask=torch.ones(pfs.shape[0], dtype=torch.float32),
            reco_jet_pt=self.reco_jet_pts[idx],
            gen_tau_label=self.gen_tau_labels[idx],
            gen_tau_pt=self.gen_tau_pts[idx]
        )


# class SimpleDNNTauBuilder:
#     def __init__(self, cfg: DictConfig, verbosity: int = 0):