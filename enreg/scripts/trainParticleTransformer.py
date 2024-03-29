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
from enreg.tools.data_management.features import FeatureStandardization
from enreg.tools.data_management.particleTransformer_dataset import ParticleTransformerDataset
from enreg.tools.models.logTrainingProgress import logTrainingProgress, logTrainingProgress_regression

import time


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
    is_energy_regression,
):
    print("::::: TRAIN LOOP :::::")
    num_jets_train = len(dataloader_train.dataset)
    loss_train = 0.0
    normalization = 0.0
    if not is_energy_regression:
        accuracy_train = 0.0
        accuracy_normalization_train = 0.0
        class_true_train = []
        class_pred_train = []
    else:
        ratios = []
        # reco_gen_ratios = []
    #     mean_absolute_errors = []
    #     mean_squared_errors = []
    #     root_mean_squared_errors = []
    #     root_mean_squared_log_errors []
    #     mae_loss = nn.L1Loss()
    weights_train = []
    print("Training model")
    model.train()
    print("Finished training model")
    for idx_batch, (X, y, weight) in enumerate(dataloader_train):
        # Compute prediction and loss
        if transform:
            X = transform(X)
        x = X["x"].to(device=dev)
        v = X["v"].to(device=dev)
        mask = X["mask"].to(device=dev)
        y = y.to(device=dev)
        weight = weight.to(device=dev)

        if is_energy_regression:
            # Predict the correction, not the full visible energy
            # jet_energy = X["reco_jet_energy"].to(device=dev)
            # pred *= jet_energy
            # pred += jet_energy
            pred = model(x, v, mask).to(device=dev)[:,0]
            predicted_pt = torch.exp(pred) * X["reco_jet_pt"].to(device=dev)
            y_for_loss = torch.log(y / X["reco_jet_pt"].to(device=dev))
        else:
            pred = model(x, v, mask).to(device=dev)
            # pred = torch.softmax(pred, dim=1)
            pred = torch.softmax(pred, dim=0)
            y_for_loss = y
        loss = loss_fn(pred, y_for_loss)
        if use_per_jet_weights:
            loss = loss * weight
        loss_train += loss.sum().item()
        normalization += torch.flatten(loss).size(dim=0)
        if not is_energy_regression:
            accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
            accuracy_train += accuracy.sum().item()
            accuracy_normalization_train += torch.flatten(accuracy).size(dim=0)
            class_true_train.extend(y.detach().cpu().numpy())
            class_pred_train.extend(pred.argmax(dim=1).detach().cpu().numpy())
        else:
            ratios.extend((predicted_pt/y).detach().cpu().numpy())

        weights_train.extend(weight.detach().cpu().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        lr_scheduler.step()
        batchsize = pred.size(dim=0)
        num_jets_processed = min((idx_batch + 1) * batchsize, num_jets_train)
        if (idx_batch % 100) == 0 or num_jets_processed >= (num_jets_train - batchsize):
            print(" Running loss: %1.6f  [%i/%s]" % (loss.mean().item(), num_jets_processed, num_jets_train))

    loss_train /= normalization
    if not is_energy_regression:
        accuracy_train /= accuracy_normalization_train
        logTrainingProgress(
            tensorboard,
            idx_epoch,
            "train",
            loss_train,
            accuracy_train,
            np.array(class_true_train),
            np.array(class_pred_train),
            np.array(weights_train),
        )
    else:
        mean_reco_gen_ratio = np.mean(np.abs(ratios))
        median_reco_gen_ratio = np.median(np.abs(ratios))
        stdev_reco_gen_ratio = np.std(np.abs(ratios))
        iqr_reco_gen_ratio = np.quantile(np.abs(ratios), 0.75) - np.quantile(np.abs(ratios), 0.25)
        logTrainingProgress_regression(
            tensorboard,
            idx_epoch,
            "train",
            loss_train,
            mean_reco_gen_ratio,
            median_reco_gen_ratio,
            stdev_reco_gen_ratio,
            iqr_reco_gen_ratio,
            np.array(weights_train),
        )

    return loss_train


def validation_loop(
    idx_epoch,
    dataloader_validation,
    transform,
    model,
    dev,
    loss_fn,
    use_per_jet_weights,
    tensorboard,
    is_energy_regression,
):
    print("::::: VALIDATION LOOP :::::")
    loss_validation = 0.0
    normalization = 0.0
    if not is_energy_regression:
        accuracy_validation = 0.0
        accuracy_normalization_validation = 0.0
        class_true_validation = []
        class_pred_validation = []
    else:
        ratios = []
    weights_validation = []
    model.eval()
    with torch.no_grad():
        for idx_batch, (X, y, weight) in enumerate(dataloader_validation):
            if transform:
                X = transform(X)
            x = X["x"].to(device=dev)
            v = X["v"].to(device=dev)
            mask = X["mask"].to(device=dev)
            y = y.to(device=dev)
            weight = weight.to(device=dev)

            if is_energy_regression:
                pred = model(x, v, mask).to(device=dev)[:,0]
                # Predict the correction, not the full visible energy
                # jet_energy = X["reco_jet_energy"].to(device=dev)
                # pred *= jet_energy
                # pred += jet_energy
                predicted_pt = torch.exp(pred) * X["reco_jet_pt"].to(device=dev)
                y_for_loss = torch.log(y / X["reco_jet_pt"].to(device=dev))
            else:
                pred = model(x, v, mask).to(device=dev)
                # pred = torch.softmax(pred, dim=1)
                pred = torch.softmax(pred, dim=0)
                y_for_loss = y

            loss = loss_fn(pred, y_for_loss)
            if use_per_jet_weights:
                loss = loss * weight
            loss_validation += loss.sum().item()
            normalization += torch.flatten(loss).size(dim=0)
            if not is_energy_regression:
                accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
                accuracy_validation += accuracy.sum().item()
                accuracy_normalization_validation += torch.flatten(accuracy).size(dim=0)

                class_true_validation.extend(y.detach().cpu().numpy())
                class_pred_validation.extend(pred.argmax(dim=1).detach().cpu().numpy())
                weights_validation.extend(weight.detach().cpu().numpy())
            else:
                ratios.extend((predicted_pt/y).detach().cpu().numpy())

    loss_validation /= normalization

    if not is_energy_regression:
        accuracy_validation /= accuracy_normalization_validation
        logTrainingProgress(
            tensorboard,
            idx_epoch,
            "validation",
            loss_validation,
            accuracy_validation,
            np.array(class_true_validation),
            np.array(class_pred_validation),
            np.array(weights_validation),
        )
    else:
        mean_reco_gen_ratio = np.mean(np.abs(ratios))
        median_reco_gen_ratio = np.median(np.abs(ratios))
        stdev_reco_gen_ratio = np.std(np.abs(ratios))
        iqr_reco_gen_ratio = np.quantile(np.abs(ratios), 0.75) - np.quantile(np.abs(ratios), 0.25)
        logTrainingProgress_regression(
            tensorboard,
            idx_epoch,
            "validation",
            loss_validation,
            mean_reco_gen_ratio,
            median_reco_gen_ratio,
            stdev_reco_gen_ratio,
            iqr_reco_gen_ratio,
            np.array(weights_validation),
        )

    return loss_validation


def run_command(cmd):
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, universal_newlines=True)
    print(result.stdout)


@hydra.main(config_path="../config", config_name="model_training", version_base=None)
def trainParticleTransformer(cfg: DictConfig) -> None:
    print("<trainParticleTransformer>:")
    is_energy_regression = cfg.models.ParticleTransformer.training.type == 'regression'

    validation_paths = []
    train_paths = []
    # if is_energy_regression:
    #     for sample in cfg.samples_to_use:
    #         train_dir = os.path.join(cfg.PT_tauID_ntuple_dir, "train", sample)
    #         train_paths.extend(glob.glob(os.path.join(train_dir, "*"))[:cfg.models.ParticleTransformer.training.max_num_files])
    #         validation_dir = os.path.join(cfg.PT_tauID_ntuple_dir, "validation", sample)
    #         validation_paths.extend(glob.glob(os.path.join(validation_dir, "*"))[:cfg.models.ParticleTransformer.training.max_num_files])
    # else:
    for sample in cfg.samples_to_use:
        print(sample)
        train_paths.extend(cfg.datasets.train[sample][:cfg.models.ParticleTransformer.training.max_num_files])
        validation_paths.extend(cfg.datasets.validation[sample][:cfg.models.ParticleTransformer.training.max_num_files])


    training_data = g.load_all_data(train_paths, n_files=-1)
    dataset_train = ParticleTransformerDataset(
        data=training_data,
        cfg=cfg.models.ParticleTransformer.dataset,
        is_energy_regression=is_energy_regression
    )
    validation_data = g.load_all_data(validation_paths, n_files=-1)
    dataset_validation = ParticleTransformerDataset(
        data=validation_data,
        cfg=cfg.models.ParticleTransformer.dataset,
        is_energy_regression=cfg.models.ParticleTransformer.training.type == 'regression'
    )

    if not is_energy_regression:
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
    print("Using device: %s" % dev)

    input_dim = 7
    if cfg.models.ParticleTransformer.dataset.use_pdgId:
        input_dim += 6
    if cfg.models.ParticleTransformer.dataset.use_lifetime:
        input_dim += 4

    print("Building model...")
    model = ParticleTransformer(
        input_dim=input_dim,
        num_classes=1 if is_energy_regression else 2,
        use_pre_activation_pair=False,
        for_inference=False,
        use_amp=False,
        metric='eta-phi', # TODO: Check this later
        verbosity=cfg.verbosity,
    ).to(device=dev)
    initWeights(model)
    print("Finished building model:")
    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_weights = sum([np.prod(p.size()) for p in model_params])
    print("#trainable parameters = %i" % num_trainable_weights)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.models.ParticleTransformer.training.batch_size,
        num_workers=cfg.models.ParticleTransformer.training.num_dataloader_workers,
        shuffle=True
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=cfg.models.ParticleTransformer.training.batch_size,
        num_workers=cfg.models.ParticleTransformer.training.num_dataloader_workers,
        shuffle=True
    )

    transform = None
    if cfg.models.ParticleTransformer.feature_standardization.standardize_inputs:
        transform = FeatureStandardization(
            method=cfg.models.ParticleTransformer.feature_standardization.method,
            features=["x", "v"],
            feature_dim=1,
            verbosity=cfg.verbosity,
        )
        transform.compute_params(dataloader_train)
        transform.save_params(cfg.models.ParticleTransformer.feature_standardization.path)

    # Actually in regression I think this is not needed, maybe only for different samples differently
    if cfg.models.ParticleTransformer.lrfinder.use_class_weights:
        classweight_bgr = cfg.models.ParticleTransformer.lrfinder.classweight_bgr
        classweight_sig = cfg.models.ParticleTransformer.lrfinder.classweight_sig
    classweight_tensor = torch.tensor([classweight_bgr, classweight_sig], dtype=torch.float32).to(device=dev)
    # loss_fn = None
    if not is_energy_regression:
        if cfg.models.ParticleTransformer.training.use_focal_loss:
            print("Using FocalLoss.")
            loss_fn = FocalLoss(
                gamma=cfg.models.ParticleTransformer.training.focal_loss_gamma,
                alpha=classweight_tensor,
                reduction="none"
            )
        else:
            print("Using CrossEntropyLoss.")
            loss_fn = nn.CrossEntropyLoss(weight=classweight_tensor, reduction="none")
    else:
        loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)

    # base_optimizer = None
    # TODO: Do we event want to change the optimizer?
    if cfg.models.ParticleTransformer.training.fast_optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    elif cfg.models.ParticleTransformer.training.fast_optimizer == "RAdam":
        base_optimizer = torch.optim.RAdam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    else:
        raise RuntimeError("Invalid configuration parameter 'fast_optimizer' !!")
    if cfg.models.ParticleTransformer.training.slow_optimizer == "Lookahead":
        print("Using %s optimizer with Lookahead." % cfg.models.ParticleTransformer.training.fast_optimizer)
        optimizer = Lookahead(base_optimizer=base_optimizer, k=10, alpha=0.5)
    elif cfg.models.ParticleTransformer.training.slow_optimizer == "None":
        print("Using %s optimizer." % cfg.models.ParticleTransformer.training.fast_optimizer)
        optimizer = base_optimizer
    else:
        raise RuntimeError("Invalid configuration parameter 'slow_optimizer' !!")

    num_batches_train = len(dataloader_train)
    print("Training for %i epochs." % cfg.models.ParticleTransformer.training.num_epochs)
    print("#batches(train) = %i" % num_batches_train)
    lr_scheduler = OneCycleLR(
        base_optimizer,
        max_lr=1.0e-3,
        epochs=cfg.models.ParticleTransformer.training.num_epochs,
        steps_per_epoch=num_batches_train,
        anneal_strategy="cos"
    )

    print("Starting training...")
    print(" current time:", datetime.datetime.now())
    tensorboard = SummaryWriter(os.path.join(cfg.output_dir, f"tensorboard_{datetime.datetime.now()}"))
    min_loss_validation = -1.0
    # early_stopper = EarlyStopper(patience=100, min_delta=0.01)
    for idx_epoch in range(cfg.models.ParticleTransformer.training.num_epochs):
        print("Processing epoch #%i" % idx_epoch)
        print(" current time:", datetime.datetime.now())

        train_loop(
            idx_epoch,
            dataloader_train,
            transform,
            model,
            dev,
            loss_fn,
            cfg.models.ParticleTransformer.training.use_per_jet_weights,
            optimizer,
            lr_scheduler,
            tensorboard,
            is_energy_regression
        )
        print(" lr = %1.3e" % lr_scheduler.get_last_lr()[0])
        # print(" lr = %1.3e" % get_lr(optimizer))
        tensorboard.add_scalar("lr", lr_scheduler.get_last_lr()[0], idx_epoch)

        loss_validation = validation_loop(
            idx_epoch,
            dataloader_validation,
            transform,
            model,
            dev,
            loss_fn,
            cfg.models.ParticleTransformer.training.use_per_jet_weights,
            tensorboard,
            is_energy_regression
        )


        if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
            print("Found new best model :)")
            best_model_file = cfg.models.ParticleTransformer.training.model_file.replace(".pt", "_best.pt")
            print("Saving best model to file %s" % best_model_file)
            best_model_output_path = os.path.join(cfg.output_dir, best_model_file)
            torch.save(model.state_dict(), best_model_output_path)
            print("Done.")
            min_loss_validation = loss_validation
        print("System utilization:")
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=1)
        print(" CPU-Util = %1.2f%%" % cpu_percent)
        print(" Memory-Usage = %i Mb" % (process.memory_info().rss / 1048576))
        if dev == "cuda":
            print("GPU:")
            run_command("nvidia-smi --id=%i" % torch.cuda.current_device())
        else:
            print("GPU: N/A")
        # if early_stopper.early_stop(loss_validation):
        #     break
    print("Finished training.")
    print("Current time:", datetime.datetime.now())

    model_output_path = os.path.join(cfg.output_dir, cfg.models.ParticleTransformer.training.model_file)
    print("Saving model to file %s" % model_output_path)
    torch.save(model.state_dict(), model_output_path)
    print("Done.")

    tensorboard.close()


if __name__ == "__main__":
    trainParticleTransformer()
