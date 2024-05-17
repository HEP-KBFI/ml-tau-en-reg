#!/usr/bin/python3

import os
import glob
import json
import yaml
import hydra
import psutil
import datetime
import subprocess
import numpy as np
from omegaconf import DictConfig
import tqdm
import sklearn
import sklearn.metrics
import awkward as ak

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from enreg.tools import general as g
from enreg.tools.losses.FocalLoss import FocalLoss
from enreg.tools.models.Lookahead import Lookahead
from enreg.tools.losses.initWeights import initWeights

from enreg.tools.models.ParticleTransformer import ParticleTransformer
from enreg.tools.models.SimpleDNN import DeepSet
from enreg.tools.models.LorentzNet import LorentzNet

from enreg.tools.data_management.features import FeatureStandardization

from enreg.tools.data_management.particleTransformer_dataset import ParticleTransformerDataset
from enreg.tools.data_management.lorentzNet_dataset import LorentzNetDataset
from enreg.tools.data_management.simpleDNN_dataset import DeepSetDataset

from enreg.tools.models.logTrainingProgress import logTrainingProgress, logTrainingProgress_regression
from enreg.tools.models.logTrainingProgress import logTrainingProgress_decaymode

import time

def unpack_ParticleTransformer_data(X, dev):
    cand_features = X["cand_features"].to(device=dev)
    cand_kinematics = X["cand_kinematics"].to(device=dev)
    mask = X["mask"].to(device=dev)
    return cand_features, cand_kinematics, mask

def unpack_LorentzNet_data(X, dev):
    cand_kinematics = X["cand_kinematics"].to(device=dev)
    beam_kinematics = X["beam_kinematics"].to(device=dev)
    kinematics = torch.swapaxes(torch.concatenate([beam_kinematics, cand_kinematics], axis=-1), 1, 2)

    cand_features = X["cand_features"].to(device=dev)
    beam_features = X["beam_features"].to(device=dev)
    scalars = torch.swapaxes(torch.concatenate([beam_features, cand_features], axis=-1), 1, 2)

    cand_mask = X["mask"].to(device=dev)
    beam_mask = X["beam_mask"].to(device=dev)
    mask = torch.swapaxes(torch.concatenate([beam_mask, cand_mask], axis=-1), 1, 2)
    return kinematics, scalars, mask

def unpack_SimpleDNN_data(X, dev):
    cand_kinematics = torch.swapaxes(X["cand_kinematics"].to(device=dev), 1, 2)
    cand_features = torch.swapaxes(X["cand_features"].to(device=dev), 1, 2)
    cand_mask = torch.swapaxes(X["mask"].to(device=dev), 1, 2)
    pfs = torch.concatenate([cand_kinematics, cand_features], axis=-1)
    return pfs, cand_mask

dataset_unpackers = {
    "ParticleTransformer": unpack_ParticleTransformer_data,
    "LorentzNet": unpack_LorentzNet_data,
    "SimpleDNN": unpack_SimpleDNN_data,
}

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_loop(
    idx_epoch,
    dataloader_train,
    transform,
    model,
    dev,
    loss_fn,
    use_per_jet_weights,
    optimizer,
    lr_scheduler,
    tensorboard,
    dataset_unpacker,
    num_classes,
    kind="jet_regression",
    train=True,
):
    if train:
        print("::::: TRAIN LOOP :::::")
    else:
        print("::::: VALID LOOP :::::")

    num_jets_train = len(dataloader_train.dataset)
    loss_train = 0.0
    normalization = 0.0

    if kind == "binary_classification":
        accuracy_train = 0.0
        accuracy_normalization_train = 0.0
        class_true_train = []
        class_pred_train = []
    elif kind == "jet_regression":
        ratios = []
    elif kind == "dm_multiclass":
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    weights_train = []

    if train:
        print("Setting model to train mode")
        model.train()
        tensorboard_tag = "train"
    else:
        print("Setting model to eval mode")
        model.eval()
        tensorboard_tag = "valid"

    for idx_batch, (X, y, weight) in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
        # Compute prediction and loss
        if transform:
            X = transform(X)
        model_inputs = dataset_unpacker(X, dev)
        y_for_loss = y[kind].to(device=dev)
        weight = weight.to(device=dev)

        if kind == "jet_regression":
            pred = model(*model_inputs).to(device=dev)[:,0]
        elif kind == "dm_multiclass":
            pred = model(*model_inputs).to(device=dev)
            y_for_loss = torch.nn.functional.one_hot(y_for_loss, num_classes).float()
        elif kind == "binary_classification":
            pred = model(*model_inputs).to(device=dev)
        loss = loss_fn(pred, y_for_loss)
        if use_per_jet_weights:
            loss = loss * weight
        loss_train += loss.sum().item()
        normalization += torch.flatten(loss).size(dim=0)

        if kind == "binary_classification":
            accuracy = (pred.argmax(dim=-1) == y_for_loss).type(torch.float32)
            accuracy_train += accuracy.sum().item()
            accuracy_normalization_train += torch.flatten(accuracy).size(dim=0)
            class_true_train.extend(y_for_loss.detach().cpu().numpy())
            class_pred_train.extend(pred.detach().cpu().numpy())
        elif kind == "jet_regression":
            pred_jet_pt = torch.exp(pred.detach().cpu())*torch.squeeze(y["reco_jet_pt"], axis=-1)
            gen_tau_pt = torch.squeeze(y["gen_tau_pt"], axis=-1)
            ratio = (pred_jet_pt/gen_tau_pt).numpy()
            ratio[np.isinf(ratio)] = 0
            ratio[np.isnan(ratio)] = 0
            ratios.extend(ratio)
        elif kind == "dm_multiclass":
            pred_dm = torch.argmax(pred.detach().cpu(), axis=-1).numpy()
            true_dm = torch.argmax(y_for_loss.cpu(), axis=-1).numpy()
            confusion_matrix += sklearn.metrics.confusion_matrix(true_dm, pred_dm, labels=range(num_classes))

        weights_train.extend(weight.detach().cpu().numpy())

        # Backpropagation
        if train:
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()

    loss_train /= normalization
    if kind == "binary_classification":
        accuracy_train /= accuracy_normalization_train
        logTrainingProgress(
            tensorboard,
            idx_epoch,
            tensorboard_tag,
            loss_train,
            accuracy_train,
            np.array(class_true_train),
            np.array(class_pred_train),
            np.array(weights_train),
        )
    elif kind == "jet_regression":
        mean_reco_gen_ratio = np.mean(np.abs(ratios))
        median_reco_gen_ratio = np.median(np.abs(ratios))
        stdev_reco_gen_ratio = np.std(np.abs(ratios))
        iqr_reco_gen_ratio = np.quantile(np.abs(ratios), 0.75) - np.quantile(np.abs(ratios), 0.25)
        logTrainingProgress_regression(
            tensorboard,
            idx_epoch,
            tensorboard_tag,
            loss_train,
            mean_reco_gen_ratio,
            median_reco_gen_ratio,
            stdev_reco_gen_ratio,
            iqr_reco_gen_ratio,
            np.array(weights_train),
            np.array(ratios)
        )
    elif kind == "dm_multiclass":
        logTrainingProgress_decaymode(
            tensorboard,
            idx_epoch,
            tensorboard_tag,
            loss_train,
            np.array(weights_train),
            confusion_matrix
        )
    tensorboard.flush()
    print("Loss = {}".format(loss_train))
    return loss_train


def run_command(cmd):
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, universal_newlines=True)
    print(result.stdout)


@hydra.main(config_path="../config", config_name="model_training", version_base=None)
def trainModel(cfg: DictConfig) -> None:

    #because we are doing plots for tensorboard, we don't want anything to crash
    plt.switch_backend('agg')

    print("<trainModel>:")

    kind = cfg.training_type
    model_config = cfg.models[cfg.model_type]
    DatasetClass = ParticleTransformerDataset

    model_output_path = os.path.join(
        cfg.output_dir,
        cfg.training_type,
        cfg.model_type,
        # when running many jobs on the cluster, it's easier to not have a bunch of different dates for each model
        # str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    )

    if os.path.isdir(model_output_path):
        raise Exception("Output directory exists: {}".format(model_output_path))

    data = g.load_all_data([os.path.join(cfg.data_path, samp) for samp in cfg.training_samples])

    #shuffle input samples
    perm = np.random.permutation(len(data))
    data_reshuf = data[perm]

    #take train and validation split
    ntrain = int(len(data_reshuf)*cfg.fraction_train)
    nvalid = int(len(data_reshuf)*cfg.fraction_valid)
    training_data = data_reshuf[:ntrain]
    validation_data = data_reshuf[ntrain:ntrain+nvalid]
    print("train={} validation={}".format(len(training_data), len(validation_data)))
    
    dataset_train = DatasetClass(
        data=training_data,
        cfg=cfg.dataset,
    )
    dataset_validation = DatasetClass(
        data=validation_data,
        cfg=cfg.dataset,
    )

    if kind == "binary_classification":
        training_targets = training_data.gen_jet_tau_decaymode == -1
        validation_targets = validation_data.gen_jet_tau_decaymode == -1
        if sum(training_targets) == 0 or sum(training_targets) == len(training_targets):
            raise AssertionError((
                "Training dataset should contain both signal and background samples."
                f"Currently #Signal = {len(training_targets) - sum(training_targets)} and #BKG = {sum(training_targets)}"
            ))
        if sum(validation_targets) == 0 or sum(validation_targets) == len(validation_targets):
            raise AssertionError((
                "Validation dataset should contain both signal and background samples."
                f"Currently #Signal = {len(validation_targets) - sum(validation_targets)} and #BKG = {sum(validation_targets)}"
            ))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(dev))

    print("Building model...")
    input_dim = model_config.input_dim
    if cfg.dataset.use_lifetime:
        input_dim += 4
    num_classes = cfg.num_classes[kind]
    if cfg.model_type == "ParticleTransformer":
        model = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric='eta-phi',
            verbosity=cfg.verbosity,
        ).to(device=dev)
    elif cfg.model_type == "LorentzNet":
        model = LorentzNet(
            n_scalar=input_dim,
            n_hidden=cfg.models.LorentzNet.hyperparameters.n_hidden,
            n_class=num_classes,
            dropout=cfg.models.LorentzNet.hyperparameters.dropout,
            n_layers=cfg.models.LorentzNet.hyperparameters.n_layers,
            c_weight=cfg.models.LorentzNet.hyperparameters.c_weight,
            verbosity=cfg.verbosity,
        ).to(device=dev)
    elif cfg.model_type == "SimpleDNN":
        model = DeepSet(input_dim, num_classes).to(device=dev)

    initWeights(model)
    print("Finished building model:")
    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_weights = sum([np.prod(p.size()) for p in model_params])
    print("#trainable parameters = {}".format(num_trainable_weights))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_dataloader_workers,
        prefetch_factor=10,
        shuffle=True
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_dataloader_workers,
        prefetch_factor=10,
        shuffle=True
    )

    transform = None
    if cfg.feature_standardization.standardize_inputs:
        transform = FeatureStandardization(
            method=cfg.feature_standardization.method,
            features=ccfg.feature_standardization.features,
            feature_dim=1,
            verbosity=cfg.verbosity,
        )
        transform.compute_params(dataloader_train)
        transform.save_params(os.path.join(model_output_path, "feature_transform.json"))

    if kind == "binary_classification":
        if cfg.training.use_class_weights:
            classweight_bgr = cfg.training.classweight_bgr
            classweight_sig = cfg.training.classweight_sig
        classweight_tensor = torch.tensor([classweight_bgr, classweight_sig], dtype=torch.float32).to(device=dev)

        if cfg.training.use_focal_loss:
            print("Using FocalLoss.")
            loss_fn = FocalLoss(
                gamma=cfg.training.focal_loss_gamma,
                alpha=classweight_tensor,
                reduction="none"
            )
        else:
            print("Using CrossEntropyLoss.")
            loss_fn = nn.CrossEntropyLoss(weight=classweight_tensor, reduction="none")
    elif kind == "jet_regression":
        loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
    elif kind == "dm_multiclass":
        loss_fn = nn.CrossEntropyLoss(reduction="none")

    if cfg.training.fast_optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    elif cfg.training.fast_optimizer == "RAdam":
        base_optimizer = torch.optim.RAdam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    else:
        raise RuntimeError("Invalid configuration parameter 'fast_optimizer' !!")
    if cfg.training.slow_optimizer == "Lookahead":
        print("Using {} optimizer with Lookahead.".format(cfg.training.fast_optimizer))
        optimizer = Lookahead(base_optimizer=base_optimizer, k=10, alpha=0.5)
    elif cfg.training.slow_optimizer == "None":
        print("Using {} optimizer.".format(cfg.training.fast_optimizer))
        optimizer = base_optimizer
    else:
        raise RuntimeError("Invalid configuration parameter 'slow_optimizer' !!")

    num_batches_train = len(dataloader_train)
    print("Training for {} epochs.".format(cfg.training.num_epochs))
    print("#batches(train) = {}".format(num_batches_train))
    lr_scheduler = OneCycleLR(
        base_optimizer,
        max_lr=1.0e-3,
        epochs=cfg.training.num_epochs,
        steps_per_epoch=num_batches_train,
        anneal_strategy="cos"
    )

    print("Starting training...")
    print(" current time:", datetime.datetime.now())
    tensorboard = SummaryWriter(os.path.join(model_output_path,"tensorboard_logs"))
    min_loss_validation = -1.0
    # early_stopper = EarlyStopper(patience=100, min_delta=0.01)
    best_model_output_path = None
    for idx_epoch in range(cfg.training.num_epochs):
        print("Processing epoch #%i" % idx_epoch)
        print(" current time:", datetime.datetime.now())

        train_loop(
            idx_epoch,
            dataloader_train,
            transform,
            model,
            dev,
            loss_fn,
            cfg.training.use_per_jet_weights,
            optimizer,
            lr_scheduler,
            tensorboard,
            dataset_unpackers[cfg.model_type],
            num_classes,
            kind=kind
        )
        print("lr = {}".format(lr_scheduler.get_last_lr()[0]))
        tensorboard.add_scalar("lr", lr_scheduler.get_last_lr()[0], idx_epoch)

        with torch.no_grad():
            loss_validation = train_loop(
                idx_epoch,
                dataloader_validation,
                transform,
                model,
                dev,
                loss_fn,
                cfg.training.use_per_jet_weights,
                None,
                None,
                tensorboard,
                dataset_unpackers[cfg.model_type],
                num_classes,
                kind=kind,
                train=False
            )

        if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
            best_model_file = "model_best.pt"
            print("Saving best model to file {}".format(best_model_file))
            best_model_output_path = os.path.join(model_output_path, best_model_file)
            torch.save(model.state_dict(), best_model_output_path)
            min_loss_validation = loss_validation
        process = psutil.Process(os.getpid())
        print(" Memory-Usage = %i Mb" % (process.memory_info().rss / 1048576))
        # if early_stopper.early_stop(loss_validation):
        #     break
    print("Finished training.")
    print("Current time:", datetime.datetime.now())

    tensorboard.close()

    print("Loading best state from {}".format(best_model_output_path))
    model.load_state_dict(torch.load(best_model_output_path))
    model.eval()

    print("Evaluating on test samples")
    for test_sample in cfg.test_samples:
        print("Evaluating on {}".format(test_sample))
        data = g.load_all_data([str(os.path.join(cfg.data_path, test_sample))])
        dataset_full = DatasetClass(
            data=data,
            cfg=cfg.dataset,
            do_preselection=False #for evaluation, ensure we do no selection
        )
        dataloader_full = DataLoader(
            dataset_full,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_dataloader_workers,
            prefetch_factor=100,
            shuffle=False
        )
        preds = []
        targets = []
        for (X, y, weight) in tqdm.tqdm(dataloader_full, total=len(dataloader_full)):
            model_inputs = dataset_unpackers[cfg.model_type](X, dev)
            y_for_loss = y[kind]
            with torch.no_grad():
                if kind == "jet_regression":
                    pred = model(*model_inputs)[:, 0]
                elif kind == "dm_multiclass":
                    pred = model(*model_inputs)
                    pred = torch.argmax(pred, axis=-1)
                elif kind == "binary_classification":
                    pred = model(*model_inputs)[:, 1]
            preds.extend(pred.detach().cpu().numpy())
            targets.extend(y_for_loss.detach().cpu().numpy())
        preds = np.array(preds)
        targets = np.array(targets)
        ak.to_parquet(ak.Record({kind: {"pred": preds, "target": targets}}), os.path.join(model_output_path, test_sample))


if __name__ == "__main__":
    trainModel()
