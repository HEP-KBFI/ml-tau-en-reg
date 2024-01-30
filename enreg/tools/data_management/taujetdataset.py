import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os
import os.path as osp
from glob import glob
import vector
import random
import yaml
import multiprocessing
import sys


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


class TauJetDataset(Dataset):
    def __init__(self, processed_dir="", filelist=[]):

        self._processed_dir = processed_dir

        self.filelist = filelist
        random.shuffle(self.filelist)

        # The order of features in the jet feature tensor
        self.reco_jet_features = ["x", "y", "z", "tau", "pt", "eta", "phi", "e"]

        # The order of features in the PF feature tensor
        self.pf_features = [
            "x",
            "y",
            "z",
            "tau",
            "pt",
            "eta",
            "phi",
            "e",
            "charge",
            "is_ch_had",
            "is_n_had",
            "is_gamma",
            "is_ele",
            "is_mu",
        ]

        self.pf_extras = [
            "reco_cand_dxy",
            "reco_cand_dz",
            "reco_cand_signed_dxy",
            "reco_cand_signed_dz",
            "reco_cand_signed_d3",
            "reco_cand_d3",
            "reco_cand_d0",
            "reco_cand_z0",
            "reco_cand_PCA_x",
            "reco_cand_PCA_y",
            "reco_cand_PCA_z",
            "reco_cand_PV_x",
            "reco_cand_PV_y",
            "reco_cand_PV_z",
            "reco_cand_dxy_err",
            "reco_cand_dz_err",
            "reco_cand_d3_err",
            "reco_cand_d0_err",
            "reco_cand_z0_err",
            "reco_cand_PCA_x_err",
            "reco_cand_PCA_y_err",
            "reco_cand_PCA_z_err",
        ]

    @property
    def raw_file_names(self):
        return self.filelist

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted(proc_list)

    @property
    def processed_dir(self):
        return self._processed_dir

    def __len__(self):
        return len(self.processed_file_names)

    def get_jet_features(self, data: ak.Record) -> torch.Tensor:
        jets = {}
        for k in data["reco_jet_p4s"].fields:
            jets[k] = data["reco_jet_p4s"][k]

        jetP4 = vector.awk(
            ak.zip(
                {
                    "px": data["reco_jet_p4s"].x,
                    "py": data["reco_jet_p4s"].y,
                    "pz": data["reco_jet_p4s"].z,
                    "mass": data["reco_jet_p4s"].tau,
                }
            )
        )
        jets["pt"] = jetP4.pt
        jets["eta"] = jetP4.eta
        jets["phi"] = jetP4.phi
        jets["e"] = jetP4.energy

        # collect jet features in a specific order to an (Njet x Nfeatjet) torch tensor
        jet_feature_tensors = []
        for feat in self.reco_jet_features:
            jet_feature_tensors.append(torch.tensor(jets[feat], dtype=torch.float32))
        jet_features = torch.stack(jet_feature_tensors, axis=-1)
        return jet_features.to(dtype=torch.float32)

    def get_pf_features(self, jet_features: torch.Tensor, data: ak.Record) -> (torch.Tensor, torch.Tensor):
        pfs = {}
        for k in data["reco_cand_p4s"].fields:
            pfs[k] = data["reco_cand_p4s"][k]
        pfs["charge"] = data["reco_cand_charge"]
        pfs["pdg"] = np.abs(data["reco_cand_pdg"])

        pfP4 = vector.awk(
            ak.zip(
                {
                    "px": data["reco_cand_p4s"].x,
                    "py": data["reco_cand_p4s"].y,
                    "pz": data["reco_cand_p4s"].z,
                    "mass": data["reco_cand_p4s"].tau,
                }
            )
        )
        pfs["pt"] = pfP4.pt
        pfs["eta"] = pfP4.eta
        pfs["phi"] = pfP4.phi
        pfs["e"] = pfP4.energy

        pfs["is_ch_had"] = pfs["pdg"] == 211
        pfs["is_n_had"] = pfs["pdg"] == 130
        pfs["is_gamma"] = pfs["pdg"] == 22
        pfs["is_ele"] = pfs["pdg"] == 11
        pfs["is_mu"] = pfs["pdg"] == 13

        # collect PF features in a specific order to an (Ncand x Nfeatcand) torch tensor
        pf_feature_tensors = []
        for feat in self.pf_features:
            pf_feature_tensors.append(torch.tensor(ak.flatten(pfs[feat]), dtype=torch.float32))
        for feat in self.pf_extras:
            pf_feature_tensors.append(torch.tensor(ak.flatten(data[feat]), dtype=torch.float32))
        pf_features = torch.stack(pf_feature_tensors, axis=-1)

        # create a tensor with (Ncand x 1) which assigns each PF candidate to the jet it belongs to
        # this can be treated like batch_index in downstream algos

        pf_per_jet = ak.num(pfs["pdg"], axis=1)
        pf_to_jet = torch.tensor(np.repeat(np.arange(len(jet_features)), pf_per_jet))
        return pf_features.to(dtype=torch.float32), pf_to_jet.to(dtype=torch.long)

    def process_file_data(self, data):
        # collect all jet features
        jet_features = self.get_jet_features(data)

        # collect all jet PF candidate features
        pf_features, pf_to_jet = self.get_pf_features(jet_features, data)

        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = data["gen_jet_tau_p4s"]
        gen_tau_p4 = torch.tensor(np.stack([p4.x, p4.y, p4.z, p4.tau], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]

        weights = torch.tensor(data["weight"]).to(dtype=torch.float32)

        # Data object with:
        #   - reco jet (jet_features, jet_pf_features)
        #   - jet PF candidates (jet_pf_features, pf_to_jet)
        #   - generator level target (gen_tau_decaymode, gen_tau_p4)

        ret_data = [
            Data(
                jet_features=jet_features[ijet : ijet + 1],
                jet_pf_features=pf_features[pf_to_jet == ijet],
                gen_tau_decaymode=gen_tau_decaymode[ijet : ijet + 1],
                gen_tau_p4=gen_tau_p4[ijet : ijet + 1],
                weight=weights[ijet : ijet + 1],
            )
            for ijet in range(len(jet_features))
        ]
        return ret_data

    def process_multiple_files(self, filenames, idx_file):
        datas = []
        for fn in filenames:
            data = ak.from_parquet(fn)
            x = self.process_file_data(data)
            if x is None:
                continue
            datas.append(x)

        assert len(datas) > 0
        datas = sum(datas[1:], datas[0])
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx_file))
        print("saved {} samples to {}".format(len(datas), p))
        torch.save(datas, p)

    def process(self, num_files_to_batch):
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            self.process_multiple_files(fns, idx_file)
            idx_file += 1

    def process_parallel(self, num_files_to_batch, num_proc):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(num_proc)
        pool.map(process_func, pars)
        # for p in pars:
        #     process_func(p)

    def get(self, idx):
        fn = "data_{}.pt".format(idx)
        p = osp.join(self.processed_dir, fn)
        data = torch.load(p, map_location="cpu")
        print("loaded {}, N={}".format(fn, len(data)))
        return data

    def __getitem__(self, idx):
        return self.get(idx)


if __name__ == "__main__":

    # path to dataset yaml
    infile = sys.argv[1]
    ds = os.path.basename(infile).split(".")[0]

    filelist = get_split_files(infile, ds)
    outp = "data/dataset_{}".format(ds)
    os.makedirs(outp)
    ds = TauJetDataset(outp, filelist)

    # merge 50 files, run 16 processes
    ds.process_parallel(50, 16)
