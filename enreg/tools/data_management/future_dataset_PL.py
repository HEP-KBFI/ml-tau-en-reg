import os
import math
import torch
import random
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from collections.abc import Sequence
from enreg.tools import general as g
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset, Subset


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


class FutureDataset(Dataset):
    def __init__(self, data_path: str, cfg: DictConfig):
        self.data_path = data_path
        self.cfg = cfg
        self.row_groups = self.load_row_groups()

    def load_row_groups(self) -> Sequence[RowGroup]:
        metadata = ak.metadata_from_parquet(self.data_path)
        num_row_groups = metadata["num_row_groups"]
        col_counts = metadata["col_counts"]
        return [
            RowGroup(self.data_path, row_group, col_counts[row_group])
            for row_group in range(num_row_groups)
        ]

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class IterableFutureDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        cfg: DictConfig,
        dataset_type: str,
        reco_jet_pt_cut: float = 20,
    ):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.row_groups = [d for d in self.dataset]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(
            f"There are {'{:,}'.format(self.num_rows)} jets in the {dataset_type} dataset."
        )
        self.reco_jet_pt_cut = reco_jet_pt_cut

    def stack_and_pad_features(self, features):
        pad_length = self.cfg.max_cands
        feature_tensor = np.stack(
            [
                ak.pad_none(features[feat], pad_length, clip=True)
                for feat in features.fields
            ],
            axis=-1,
        )
        feature_tensor = ak.to_numpy(ak.fill_none(feature_tensor, 0))
        # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
        feature_tensor = np.swapaxes(feature_tensor, 1, 2)
        feature_tensor[np.isnan(feature_tensor)] = 0
        feature_tensor[np.isinf(feature_tensor)] = 0
        return feature_tensor

    def build_tensors(self, data: ak.Array):
        jet_constituent_p4s = g.reinitialize_p4(data.reco_cand_p4s)
        gen_jet_tau_p4s = g.reinitialize_p4(data.gen_jet_tau_p4s)
        jet_p4s = g.reinitialize_p4(data.reco_jet_p4s)

        feature_calculations = {
            "cand_pt": lambda: jet_constituent_p4s.pt,
            "cand_mass": lambda: jet_constituent_p4s.mass,
            "cand_deta": lambda: jet_constituent_p4s.deltaeta(jet_p4s),
            "cand_dphi": lambda: jet_constituent_p4s.deltaphi(jet_p4s),
            "cand_logpt": lambda: np.log(jet_constituent_p4s.pt),
            "cand_loge": lambda: np.log(jet_constituent_p4s.energy),
            "cand_logptrel": lambda: np.log(jet_constituent_p4s.pt / jet_p4s.pt),
            "cand_logerel": lambda: np.log(jet_constituent_p4s.energy / jet_p4s.energy),
            "cand_deltaR": lambda: jet_constituent_p4s.deltaR(jet_p4s),
            "cand_charge": lambda: data.reco_cand_charge,
            "cand_isElectron": lambda: ak.values_astype(
                abs(data.reco_cand_pdg) == 11, np.float32
            ),
            "cand_isMuon": lambda: ak.values_astype(
                abs(data.reco_cand_pdg) == 13, np.float32
            ),
            "cand_isPhoton": lambda: ak.values_astype(
                abs(data.reco_cand_pdg) == 22, np.float32
            ),
            "cand_isChargedHadron": lambda: ak.values_astype(
                abs(data.reco_cand_pdg) == 211, np.float32
            ),
            "cand_isNeutralHadron": lambda: ak.values_astype(
                abs(data.reco_cand_pdg) == 130, np.float32
            ),
        }
        main_features = ak.Array(
            {feature: feature_calculations[feature]() for feature in self.cfg.features}
        )
        cand_kinematics = ak.Array(
            {
                "cand_px": jet_constituent_p4s.px,
                "cand_py": jet_constituent_p4s.py,
                "cand_pz": jet_constituent_p4s.pz,
                "cand_en": jet_constituent_p4s.energy,
            }
        )
        cand_lifetimes = ak.Array(
            {
                "cand_dz": data.reco_cand_dz,
                "cand_dz_err": data.reco_cand_dz_err,
                "cand_dxy": data.reco_cand_dxy,
                "cand_dxy_err": data.reco_cand_dxy_err,
            }
        )

        features_tensor = torch.tensor(
            self.stack_and_pad_features(main_features), dtype=torch.float32
        )
        kinematics_tensor = torch.tensor(
            self.stack_and_pad_features(cand_kinematics), dtype=torch.float32
        )
        lifetimes_tensor = torch.tensor(
            self.stack_and_pad_features(cand_lifetimes), dtype=torch.float32
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
                dtype=torch.float32,
            ),
            dim=1,
        )
        if "weight" in data.fields:
            weight_tensors = torch.tensor(ak.to_numpy(data.weight), dtype=torch.float32)
        else:
            weight_tensors = torch.tensor(
                ak.ones_like(data.gen_jet_tau_decaymode), dtype=torch.float32
            )
        reco_jet_pt = torch.tensor(ak.to_numpy(jet_p4s.pt), dtype=torch.float32)
        gen_tau_pt = torch.tensor(ak.to_numpy(gen_jet_tau_p4s.pt), dtype=torch.float32)
        jet_regression_target = torch.log(gen_tau_pt / reco_jet_pt)
        gen_jet_tau_decaymode = torch.tensor(
            ak.to_numpy(data.gen_jet_tau_decaymode)
        ).long()
        gen_jet_tau_decaymode_exists = (gen_jet_tau_decaymode != -1).long()

        # X, y, w
        return (
            # X - model inputs
            {
                "cand_kinematics": kinematics_tensor,
                "cand_features": features_tensor,
                "cand_lifetimes": lifetimes_tensor,
                "mask": node_mask_tensors,
                "reco_jet_pt": reco_jet_pt,
            },
            # y - targets
            {
                "reco_jet_pt": reco_jet_pt,
                "gen_tau_pt": gen_tau_pt,
                "jet_regression": jet_regression_target,
                "binary_classification": gen_jet_tau_decaymode_exists,
                "dm_multiclass": gen_jet_tau_decaymode,
            },
            # weights
            weight_tensors,
        )

    # def __len__(self):
    #     return self.num_rows  # TODO: IterableDataset` has `__len__` defined. In combination with multi-process data
    #     # TODO: loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured
    #     #  TODO: independently to avoid having duplicate data.

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
            for idx_jet in range(len(data)):
                yield (
                    {k: v[idx_jet] for k, v in tensors[0].items()},
                    {k: v[idx_jet] for k, v in tensors[1].items()},
                    tensors[2][idx_jet],
                )


def train_val_split_shuffle(
    concat_dataset: ConcatDataset,
    val_split: float = 0.2,
    seed: int = 42,
    max_train_jets: int = 1e6,
    row_group_size: int = 1024,
):
    total_len = len(concat_dataset)
    indices = list(range(total_len))
    random.seed(seed)
    random.shuffle(indices)

    split = int(total_len * val_split)
    if max_train_jets == -1:
        train_end_idx = None
    else:
        num_train_rows = int(np.ceil(max_train_jets / row_group_size))
        train_end_idx = split + num_train_rows
    val_indices = indices[:split]
    train_indices = indices[split:train_end_idx]

    train_subset = Subset(concat_dataset, train_indices)
    val_subset = Subset(concat_dataset, val_indices)

    return train_subset, val_subset


class FutureDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, samples: list, training_task: str):
        super().__init__()
        self.cfg = cfg
        self.samples = samples
        self.reco_jet_pt_cut = self.cfg.reco_jet_pt_cut[training_task]
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None

    def get_dataset_path(self, sample: str, dataset_type: str) -> str:
        return os.path.join(self.cfg.data_path, f"{sample}_{dataset_type}.parquet")

    def setup(self, stage: str):
        if stage == "fit":
            train_datasets = []
            for sample in self.samples:
                data_path = self.get_dataset_path(sample=sample, dataset_type="train")
                full_train_dataset = FutureDataset(
                    data_path=data_path,
                    cfg=self.cfg.dataset,
                )
                train_datasets.append(full_train_dataset)
            train_concat_dataset = ConcatDataset(train_datasets)
            train_subset, val_subset = train_val_split_shuffle(
                concat_dataset=train_concat_dataset,
                val_split=self.cfg.dataset.fraction_valid,
                max_train_jets=self.cfg.trainSize,
                row_group_size=self.cfg.dataset.row_group_size,
            )
            train_iterable_dataset = IterableFutureDataset(
                dataset=train_subset,
                cfg=self.cfg.dataset,
                dataset_type="train",
                reco_jet_pt_cut=self.reco_jet_pt_cut,
            )
            val_iterable_dataset = IterableFutureDataset(
                dataset=val_subset,
                cfg=self.cfg.dataset,
                dataset_type="validation",
                reco_jet_pt_cut=self.reco_jet_pt_cut,
            )
            self.train_loader = DataLoader(
                train_iterable_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                val_iterable_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

        elif stage == "test":
            test_datasets = []
            for sample in self.samples:
                data_path = self.get_dataset_path(sample=sample, dataset_type="train")
                test_dataset = FutureDataset(
                    data_path=data_path,
                    cfg=self.cfg.dataset,
                )
                test_datasets.append(test_dataset)
            test_concat_dataset = ConcatDataset(test_datasets)
            test_iterable_dataset = IterableFutureDataset(
                dataset=test_concat_dataset,
                cfg=self.cfg.dataset,
                dataset_type="test",
                reco_jet_pt_cut=self.reco_jet_pt_cut,
            )
            self.test_loader = DataLoader(
                test_iterable_dataset, batch_size=self.cfg.training.batch_size
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
