#!/usr/bin/python3

import datetime
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
import psutil
import subprocess
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from ParticleTransformerDataset import ParticleTransformerDataset
from ParticleTransformer import ParticleTransformer
from FeatureStandardization import FeatureStandardization
from FocalLoss import FocalLoss
from Lookahead import Lookahead
from logTrainingProgress import logTrainingProgress
from initWeights import initWeights


def get_split_files(cfg_filename, split):
    with open(cfg_filename, "r") as cfg_file:
        data = yaml.safe_load(cfg_file)
        paths = data[split]["paths"]

        # FIXME: this is hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch/persistent for GPU training
        # paths = [p.replace("/local/laurits", "./data") for p in paths]
        return paths


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
):
    num_jets_train = len(dataloader_train.dataset)
    loss_train = 0.0
    loss_normalization_train = 0.0
    accuracy_train = 0.0
    accuracy_normalization_train = 0.0
    class_true_train = []
    class_pred_train = []
    weights_train = []
    model.train()
    for idx_batch, (X, y, weight) in enumerate(dataloader_train):
        # Compute prediction and loss
        if transform:
            X = transform(X)
        x = X["x"].to(device=dev)
        v = X["v"].to(device=dev)
        mask = X["mask"].to(device=dev)
        y = y.squeeze(dim=1).to(device=dev)
        weight = weight.squeeze(dim=1).to(device=dev)
        pred = model(x, v, mask).to(device=dev)

        loss = None
        if use_per_jet_weights:
            loss = loss_fn(pred, y)
            loss = loss * weight
        else:
            loss = loss_fn(pred, y)
        loss_train += loss.sum().item()
        loss_normalization_train += torch.flatten(loss).size(dim=0)
        accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
        accuracy_train += accuracy.sum().item()
        accuracy_normalization_train += torch.flatten(accuracy).size(dim=0)

        class_true_train.extend(y.detach().cpu().numpy())
        class_pred_train.extend(pred.argmax(dim=1).detach().cpu().numpy())
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

    loss_train /= loss_normalization_train
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
):
    loss_validation = 0.0
    loss_normalization_validation = 0.0
    accuracy_validation = 0.0
    accuracy_normalization_validation = 0.0
    class_true_validation = []
    class_pred_validation = []
    weights_validation = []
    model.eval()
    with torch.no_grad():
        for idx_batch, (X, y, weight) in enumerate(dataloader_validation):
            if transform:
                X = transform(X)
            x = X["x"].to(device=dev)
            v = X["v"].to(device=dev)
            mask = X["mask"].to(device=dev)
            y = y.squeeze(dim=1).to(device=dev)
            weight = weight.squeeze(dim=1).to(device=dev)
            pred = model(x, v, mask).to(device=dev)

            if use_per_jet_weights:
                loss = loss_fn(pred, y)
                loss = loss * weight
            else:
                loss = loss_fn(pred, y).item()
            loss_validation += loss.sum().item()
            loss_normalization_validation += torch.flatten(loss).size(dim=0)
            accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
            accuracy_validation += accuracy.sum().item()
            accuracy_normalization_validation += torch.flatten(accuracy).size(dim=0)

            class_true_validation.extend(y.detach().cpu().numpy())
            class_pred_validation.extend(pred.argmax(dim=1).detach().cpu().numpy())
            weights_validation.extend(weight.detach().cpu().numpy())

    loss_validation /= loss_normalization_validation
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

    return loss_validation


def run_command(cmd):
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, universal_newlines=True)
    print(result.stdout)


@hydra.main(config_path="../config", config_name="trainParticleTransformer", version_base=None)
def trainParticleTransformer(train_cfg: DictConfig) -> None:
    print("<trainParticleTransformer>:")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    config_path = "./config"
    for cfg in hydra_cfg["runtime"]["config_sources"]:
        if cfg["schema"] == "file":
            config_path = cfg["path"]
    config_name = hydra_cfg["job"]["config_name"]
    print("Loading training configuration from file: %s/%s.yaml" % (config_path, config_name))
    print(" model_file = %s" % train_cfg.model_file)
    print(" model_config_file = %s" % train_cfg.model_config_file)
    outpath = hydra_cfg["runtime"]["output_dir"]
    print(" outpath = %s" % outpath)

    filelist_train = get_split_files("config/datasets/train.yaml", "train")
    filelist_validation = get_split_files("config/datasets/validation.yaml", "validation")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: %s" % dev)

    jsonFileName = "%s/%s" % (config_path, train_cfg.model_config_file)
    print("Loading model configuration from file: %s" % jsonFileName)
    ParticleTransformer_cfg = None
    if os.path.isfile(jsonFileName):
        jsonFile = open(jsonFileName, "r")
        ParticleTransformer_cfg = json.load(jsonFile)
        if "ParticleTransformer" not in ParticleTransformer_cfg.keys():
            raise RuntimeError("Failed to parse config file %s !!")
        ParticleTransformer_cfg = ParticleTransformer_cfg["ParticleTransformer"]
        for key, value in ParticleTransformer_cfg.items():
            print(" %s = " % key, value)
        jsonFile.close()
    else:
        raise RuntimeError("Failed to read config file %s !!")

    max_cands = ParticleTransformer_cfg["max_cands"]
    metric = ParticleTransformer_cfg["metric"]
    use_pdgId = ParticleTransformer_cfg["use_pdgId"]
    use_lifetime = ParticleTransformer_cfg["use_lifetime"]
    input_dim = 7
    if use_pdgId:
        input_dim += 6
    if use_lifetime:
        input_dim += 4
    standardize_inputs = ParticleTransformer_cfg["standardize_inputs"]
    preselection = {
        "min_jet_theta": ParticleTransformer_cfg["min_jet_theta"],
        "max_jet_theta": ParticleTransformer_cfg["max_jet_theta"],
        "min_jet_pt": ParticleTransformer_cfg["min_jet_pt"],
        "max_jet_pt": ParticleTransformer_cfg["max_jet_pt"],
    }

    print("Building model...")
    model = ParticleTransformer(
        input_dim=input_dim,
        num_classes=2,
        use_pre_activation_pair=False,
        for_inference=False,
        use_amp=False,
        metric=metric,
        verbosity=train_cfg.verbosity,
    ).to(device=dev)
    initWeights(model)
    print("Finished building model:")
    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_weights = sum([np.prod(p.size()) for p in model_params])
    print("#trainable parameters = %i" % num_trainable_weights)

    print("Starting to build training dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_train = ParticleTransformerDataset(
        filelist_train,
        max_num_files=train_cfg.max_num_files,
        max_cands=max_cands,
        use_pdgId=use_pdgId,
        use_lifetime=use_lifetime,
        preselection=preselection,
    )
    print("Finished building training dataset.")
    print(" current time:", datetime.datetime.now())

    print("Starting to build validation dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_validation = ParticleTransformerDataset(
        filelist_validation,
        max_num_files=train_cfg.max_num_files,
        metric=metric,
        max_cands=max_cands,
        use_pdgId=use_pdgId,
        use_lifetime=use_lifetime,
        preselection=preselection,
    )
    print("Finished building validation dataset.")
    print(" current time:", datetime.datetime.now())

    dataloader_train = DataLoader(
        dataset_train, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )

    transform = None
    if standardize_inputs:
        transform = FeatureStandardization(
            method=ParticleTransformer_cfg["method_FeatureStandardization"],
            features=["x", "v"],
            feature_dim=1,
            verbosity=train_cfg.verbosity,
        )
        transform.compute_params(dataloader_train)
        transform.save_params(ParticleTransformer_cfg["json_file_FeatureStandardization"])

    classweight_bgr = 1.0
    classweight_sig = 1.0
    if train_cfg.use_class_weights:
        classweight_bgr = train_cfg.classweight_bgr
        classweight_sig = train_cfg.classweight_sig
    classweight_tensor = torch.tensor([classweight_bgr, classweight_sig], dtype=torch.float32).to(device=dev)
    loss_fn = None
    if train_cfg.use_focal_loss:
        print("Using FocalLoss.")
        loss_fn = FocalLoss(gamma=train_cfg.focal_loss_gamma, alpha=classweight_tensor, reduction="none")
    else:
        print("Using CrossEntropyLoss.")
        loss_fn = nn.CrossEntropyLoss(weight=classweight_tensor, reduction="none")

    base_optimizer = None
    if train_cfg.fast_optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    elif train_cfg.fast_optimizer == "RAdam":
        base_optimizer = torch.optim.RAdam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    else:
        raise RuntimeError("Invalid configuration parameter 'fast_optimizer' !!")
    optimizer = None
    if train_cfg.slow_optimizer == "Lookahead":
        print("Using %s optimizer with Lookahead." % train_cfg.fast_optimizer)
        optimizer = Lookahead(base_optimizer=base_optimizer, k=10, alpha=0.5)
    elif train_cfg.slow_optimizer == "None":
        print("Using %s optimizer." % train_cfg.fast_optimizer)
        optimizer = base_optimizer
    else:
        raise RuntimeError("Invalid configuration parameter 'slow_optimizer' !!")

    num_batches_train = len(dataloader_train)
    print("Training for %i epochs." % train_cfg.num_epochs)
    print("#batches(train) = %i" % num_batches_train)
    lr_scheduler = OneCycleLR(
        base_optimizer, max_lr=1.0e-3, epochs=train_cfg.num_epochs, steps_per_epoch=num_batches_train, anneal_strategy="cos"
    )

    print("Starting training...")
    print(" current time:", datetime.datetime.now())
    tensorboard = SummaryWriter(outpath + "/tensorboard")
    min_loss_validation = -1.0
    for idx_epoch in range(train_cfg.num_epochs):
        print("Processing epoch #%i" % idx_epoch)
        print(" current time:", datetime.datetime.now())

        train_loop(
            idx_epoch,
            dataloader_train,
            transform,
            model,
            dev,
            loss_fn,
            train_cfg.use_per_jet_weights,
            optimizer,
            lr_scheduler,
            tensorboard,
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
            train_cfg.use_per_jet_weights,
            tensorboard,
        )
        if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
            print("Found new best model :)")
            best_model_file = train_cfg.model_file.replace(".pt", "_best.pt")
            print("Saving best model to file %s" % best_model_file)
            torch.save(model.state_dict(), best_model_file)
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
    print("Finished training.")
    print(" current time:", datetime.datetime.now())

    print("Saving model to file %s" % train_cfg.model_file)
    torch.save(model.state_dict(), train_cfg.model_file)
    print("Done.")

    tensorboard.close()


if __name__ == "__main__":
    trainParticleTransformer()
