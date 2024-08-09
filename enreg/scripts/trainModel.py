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
from enreg.tools.models.OmniParT import OmniParT

from enreg.tools.data_management.features import FeatureStandardization

from enreg.tools.data_management.particleTransformer_dataset import load_row_groups, ParticleTransformerDataset

from enreg.tools.models.logTrainingProgress import logTrainingProgress, logTrainingProgress_regression
from enreg.tools.models.logTrainingProgress import logTrainingProgress_decaymode

import time


def unpack_data(X, dev, feature_set):
    # Create a dictionary for each feature
    features_as_dict = {
        feature: X[feature].to(device=dev) for feature in feature_set
    }

    # Concatenate chosen features
    particle_features = torch.cat([features_as_dict[feat] for feat in feature_set], axis=1)

    cand_kinematics = X["cand_kinematics"].to(device=dev)
    mask = X["mask"].to(device=dev).bool()
    return particle_features, cand_kinematics, mask


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


def get_weights(train_dataset, validation_dataset, n_bins=20):
    """ Returns the weights for each of the jets based on the pT of the generated tau. The fewer events in a given
    genTau pT bin, the bigger the weight.
    """
    train_gen_tau_pt = g.reinitialize_p4(train_dataset.gen_jet_tau_p4s).pt
    val_gen_tau_pt = g.reinitialize_p4(validation_dataset.gen_jet_tau_p4s).pt
    hist, bin_edges = np.histogram(train_gen_tau_pt, bins=n_bins, range=(0, 180))
    weight_histogram = np.median(hist)/hist

    train_histo_location = np.digitize(train_gen_tau_pt, bins=bin_edges)
    val_histo_location = np.digitize(val_gen_tau_pt, bins=bin_edges)
    map_from = np.array(range(1, len(bin_edges)))
    map_to = weight_histogram

    train_mask = train_histo_location[:,None] == map_from
    val_mask = val_histo_location[:,None] == map_from

    train_values = map_to[np.argmax(train_mask, axis=1)]
    val_values = map_to[np.argmax(val_mask, axis=1)]

    train_weights = ak.Array(np.where(np.any(train_mask, axis=1), train_values, train_histo_location))
    val_weights = ak.Array(np.where(np.any(val_mask, axis=1), val_values, val_histo_location))
    return torch.tensor(train_weights), torch.tensor(val_weights)


def train_loop(
    idx_epoch,
    dataloader_train,
    model,
    dev,
    loss_fn,
    cfg,
    use_per_jet_weights,
    optimizer,
    lr_scheduler,
    tensorboard,
    feature_set,
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
        model_inputs = unpack_data(X, dev, feature_set)
        y_for_loss = y[kind].to(device=dev)
        weight = weight.to(device=dev)

        if cfg.model_type == 'OmniParT':
            if idx_epoch < cfg.models.OmniParT.num_rounds_frozen_backbone:
                frost = 'freeze'
            else:
                frost = 'unfreeze'
            model_inputs = model_inputs + (frost,)
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
    print("<trainModel>:")

    #because we are doing plots for tensorboard, we don't want anything to crash
    plt.switch_backend('agg')

    feature_set = cfg.dataset.feature_set
    print(f"Using features: {feature_set}")

    kind = cfg.training_type
    model_config = cfg.models[cfg.model_type]

    model_output_path = os.path.join(
        cfg.output_dir,
        cfg.training_type,
        cfg.model_type,
        # when running many jobs on the cluster, it's easier to not have a bunch of different dates for each model
        # str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    )

    #if we are going to train the model, ensure a model does not already exist (e.g. if doing testing as a separate step)
    if cfg.train and os.path.isdir(model_output_path):
        raise Exception("Output directory exists while train=True: {}".format(model_output_path))

    # get the number of row groups (independently loadable chunks) in each input file
    data = sum([load_row_groups(os.path.join(cfg.data_path, fn)) for fn in cfg.training_samples], [])

    #shuffle row groups
    perm = np.random.permutation(len(data))
    data_reshuf = [data[p] for p in perm]

    #take train and validation split from the row groups
    ntrain = int(len(data_reshuf)*cfg.fraction_train)
    nvalid = int(len(data_reshuf)*cfg.fraction_valid)
    training_data = data_reshuf[:ntrain]
    validation_data = data_reshuf[ntrain:ntrain+nvalid]
    print(f"row groups: train={len(training_data)} validation={len(validation_data)}")
    
    dataset_train = ParticleTransformerDataset(
        row_groups=training_data,
        cfg=cfg.dataset,
    )
    dataset_validation = ParticleTransformerDataset(
        row_groups=validation_data,
        cfg=cfg.dataset,
    )
    if kind == "jet_regression" and cfg.training.apply_regression_weights:
        weights_train, weights_validation = get_weights(dataset_train, dataset_validation)
        dataset_train.weight_tensors = weights_train
        dataset_validation.weight_tensors = weights_validation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    print("Building model...")

    #configure the number of model input dimensions based on the input features 
    input_dim = 0
    if 'cand_kinematics' in feature_set:
        input_dim += 4
    if 'cand_ParT_features' in feature_set:
        input_dim += 13
    if 'cand_lifetimes' in feature_set:
        input_dim += 4
    # TODO: other feature sets also?
    
    num_classes = cfg.num_classes[kind]
    if cfg.model_type == "ParticleTransformer":
        model = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=cfg.models.ParticleTransformer.hyperparameters.num_layers,
            embed_dims=cfg.models.ParticleTransformer.hyperparameters.embed_dims,
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
    elif cfg.model_type == "OmniParT":
        model = OmniParT(
            input_dim=input_dim,
            cfg=cfg.models.OmniParT,
            num_classes=num_classes,
            num_layers=cfg.models.OmniParT.hyperparameters.num_layers,
            embed_dims=cfg.models.OmniParT.hyperparameters.embed_dims,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric='eta-phi',
            verbosity=cfg.verbosity,
        ).to(device=dev)

    initWeights(model)
    print("Finished building model:")
    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_weights = sum([np.prod(p.size()) for p in model_params])
    print("#trainable parameters = {}".format(num_trainable_weights))
    
    best_model_output_path = os.path.join(model_output_path, "model_best.pt")

    if cfg.train:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_dataloader_workers,
            prefetch_factor=cfg.training.prefetch_factor,
        )
        dataloader_validation = DataLoader(
            dataset_validation,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_dataloader_workers,
            prefetch_factor=cfg.training.prefetch_factor,
        )

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
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
        elif cfg.training.fast_optimizer == "RAdam":
            base_optimizer = torch.optim.RAdam(model.parameters(), lr=cfg.training.lr)
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer,
            T_max=len(dataloader_train)*cfg.training.num_epochs,
            eta_min=cfg.training.lr*0.01
        )
        print("Starting training...")
        print(" current time:", datetime.datetime.now())
        tensorboard = SummaryWriter(os.path.join(model_output_path,"tensorboard_logs"))
        min_loss_validation = -1.0
        # early_stopper = EarlyStopper(patience=10)
        losses_train = []
        losses_validation = []
        for idx_epoch in range(cfg.training.num_epochs):
            print("Processing epoch #%i" % idx_epoch)
            print(" current time:", datetime.datetime.now())

            loss_train = train_loop(
                idx_epoch,
                dataloader_train,
                model,
                dev,
                loss_fn,
                cfg,
                cfg.training.use_per_jet_weights,
                optimizer,
                lr_scheduler,
                tensorboard,
                feature_set,
                num_classes,
                kind=kind,
            )
            print("lr = {}".format(lr_scheduler.get_last_lr()[0]))
            tensorboard.add_scalar("lr", lr_scheduler.get_last_lr()[0], idx_epoch)

            with torch.no_grad():
                loss_validation = train_loop(
                    idx_epoch,
                    dataloader_validation,
                    model,
                    dev,
                    loss_fn,
                    cfg,
                    cfg.training.use_per_jet_weights,
                    None,
                    None,
                    tensorboard,
                    feature_set,
                    num_classes,
                    kind=kind,
                    train=False,
                )

            losses_train.append(loss_train)
            losses_validation.append(loss_validation)

            if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
                print("Saving best model to file {}".format(best_model_output_path))
                torch.save(model.state_dict(), best_model_output_path)
                min_loss_validation = loss_validation
            process = psutil.Process(os.getpid())
            print(" Memory-Usage = %i Mb" % (process.memory_info().rss / 1048576))

            with open(os.path.join(model_output_path, "history.json"), "w") as fi:
                json.dump({"losses_train": losses_train, "losses_validation": losses_validation}, fi, indent=4)
            # if early_stopper.early_stop(loss_validation):
            #     break
        print("Finished training.")
        print("Current time:", datetime.datetime.now())

        tensorboard.close()

    if cfg.test:
        print("Loading best state from {}".format(best_model_output_path))
        model.load_state_dict(torch.load(best_model_output_path, map_location=dev))
        model.eval()

        print("Evaluating on test samples")
        for test_sample in cfg.test_samples:
            print("Evaluating on {}".format(test_sample))
            data = load_row_groups(os.path.join(cfg.data_path, test_sample))
            dataset_full = ParticleTransformerDataset(
                row_groups=data,
                cfg=cfg.dataset,
            )
            dataloader_full = DataLoader(
                dataset_full,
                batch_size=cfg.training.batch_size,
            )
            preds = []
            targets = []
            for (X, y, weight) in tqdm.tqdm(dataloader_full, total=len(dataloader_full)):
                model_inputs = unpack_data(X, dev, feature_set)
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
