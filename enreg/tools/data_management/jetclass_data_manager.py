import os
import math
import glob
import torch
import vector
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from collections.abc import Sequence
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset


def to_p4(energy, px, py, pz):
    return vector.awk(ak.zip({"energy": energy, "x": px, "y": py, "z": pz,}))


def stack_and_pad_features(cand_features, max_constituents):
    cand_features_tensors = np.stack([ak.pad_none(cand_features[feat], max_constituents, clip=True) for feat in cand_features.fields], axis=-1)
    cand_features_tensors = ak.to_numpy(ak.fill_none(cand_features_tensors, 0))
    # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
    cand_features_tensors = np.swapaxes(cand_features_tensors, 1, 2)
    cand_features_tensors[np.isnan(cand_features_tensors)] = 0
    cand_features_tensors[np.isinf(cand_features_tensors)] = 0
    return cand_features_tensors


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


class IterableJetClassDataset(IterableDataset):
    def __init__(self, data_dir: str, dataset_type: str, cfg: DictConfig):
        """ The base class for JetClass (https://zenodo.org/records/6619768) dataset"""

        self.data_paths = glob.glob(os.path.join(data_dir, dataset_type, "*.parquet"))
        self.cfg = cfg
        self.row_groups = self.load_row_groups()
        data_permutation_indices = np.random.permutation(len(self.row_groups))
        self.row_groups = [self.row_groups[p] for p in data_permutation_indices]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the {dataset_type} dataset.")

    def load_row_groups(self) -> Sequence[RowGroup]:
        all_row_groups = []
        for data_path in self.data_paths:
            metadata = ak.metadata_from_parquet(data_path)
            num_row_groups = metadata["num_row_groups"]
            col_counts = metadata["col_counts"]
            all_row_groups.extend(
                [RowGroup(data_path, row_group, col_counts[row_group]) for row_group in range(num_row_groups)]
            )
        return all_row_groups

    def build_tensors(self, data: ak.Array):
        cand_p4 = to_p4(energy=data.part_energy, px=data.part_px, py=data.part_py, pz=data.part_pz)
        jet_p4 = vector.zip({'pt': data.jet_pt, 'phi': data.jet_phi, 'eta': data.jet_eta, 'energy': data.jet_energy,})
        cand_features = ak.Array({
            'part_logpt': np.log(cand_p4.pt),
            'part_loge': np.log(data.part_energy),
            'part_deta': data.part_deta,
            'part_dphi': data.part_dphi,
            'part_logptrel': np.log(cand_p4.pt / data.jet_pt),
            'part_logerel': np.log(data.part_energy / data.jet_energy),
            'part_deltaR': cand_p4.deltaR(jet_p4),
            'part_charge': data.part_charge,
            'part_isChargedHadron': data.part_isChargedHadron,
            'part_isNeutralHadron': data.part_isNeutralHadron,
            'part_isPhoton': data.part_isPhoton,
            'part_isElectron': data.part_isElectron,
            'part_isMuon': data.part_isMuon,
        })
        cand_kinematics = ak.Array({
            "cand_px": data.part_px,
            "cand_py": data.part_py,
            "cand_pz": data.part_pz,
            "cand_en": data.part_energy,
        })
        # cand_lifetimes = ak.Array({
        #     'part_d0val': data.part_d0val,
        #     'part_d0err': data.part_d0err,
        #     'part_dzval': data.part_dzval,
        #     'part_dzerr': data.part_dzerr,
        # })
        # omnijet_features = ak.Array({
        #     'part_pt': cand_p4.pt,
        #     'part_mass': cand_p4.mass,
        #     'part_deta': data.part_deta,
        #     'part_dphi': data.part_dphi,
        #     'part_charge': data.part_charge,
        #     'part_isChargedHadron': data.part_isChargedHadron,
        #     'part_isNeutralHadron': data.part_isNeutralHadron,
        #     'part_isPhoton': data.part_isPhoton,
        #     'part_isElectron': data.part_isElectron,
        #     'part_isMuon': data.part_isMuon,
        # })
        # omnijet_cand_kinematics = ak.Array({
        #     'part_pt': cand_p4.pt,
        #     'part_deta': data.part_deta,
        #     'part_dphi': data.part_dphi,
        # })
        cand_feature_tensors = torch.tensor(stack_and_pad_features(cand_features, self.cfg.max_constituents), dtype=torch.float32)
        cand_kinematics_tensors = torch.tensor(stack_and_pad_features(cand_kinematics, self.cfg.max_constituents), dtype=torch.float32)
        # cand_lifetimes_tensors = torch.tensor(stack_and_pad_features(cand_lifetimes, self.cfg.max_constituents), dtype=torch.float32)
        # omnijet_feature_tensors = torch.tensor(stack_and_pad_features(omnijet_features, self.cfg.max_constituents), dtype=torch.float32)
        # omnijet_cand_kinematics_tensors = torch.tensor(stack_and_pad_features(omnijet_cand_kinematics, self.cfg.max_constituents), dtype=torch.float32)

        node_mask_tensors = torch.unsqueeze(
            torch.tensor(
                ak.to_numpy(ak.fill_none(ak.pad_none(ak.ones_like(data.part_isMuon), self.cfg.max_constituents, clip=True), 0,)),
                dtype=torch.bool
            ),
            dim=1
        )
        targets = ak.concatenate(
            [ak.values_astype(ak.unflatten(data[label], counts=1), int) for label in self.cfg.labels],
            axis=-1
        )
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return (
            #X - model inputs
            cand_feature_tensors,
            cand_kinematics_tensors,
            node_mask_tensors,
            #y - target
            target_tensor
        )

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            row_groups_start = worker_id*per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]

        for row_group in row_groups_to_process:
            data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
            tensors = self.build_tensors(data)

            # return individual jets from the dataset
            for ijet in range(len(data)):
                yield (
                    tensors[0][ijet],
                    tensors[1][ijet],
                    tensors[2][ijet],
                    tensors[3][ijet],
                    )


class JetClassDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.save_hyperparameters()
        super().__init__()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IterableJetClassDataset(data_dir=self.cfg.jetclass_parquet_dir, dataset_type="train", cfg=self.cfg)
            self.val_dataset = IterableJetClassDataset(data_dir=self.cfg.jetclass_parquet_dir, dataset_type="val", cfg=self.cfg)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
        elif stage == "test":
            self.test_dataset = IterableJetClassDataset(data_dir=self.cfg.jetclass_parquet_dir, dataset_type="test", cfg=self.cfg)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
