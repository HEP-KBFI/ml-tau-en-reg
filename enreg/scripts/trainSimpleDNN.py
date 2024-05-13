import os
import tqdm
import torch
import hydra
import datetime
import numpy as np
from torch import nn
from omegaconf import DictConfig
from enreg.tools import general as g
from torch.utils.data import DataLoader
from enreg.tools.models.SimpleDNN import DeepSet
from torch.utils.tensorboard import SummaryWriter
from enreg.tools.data_management.simpleDNN_dataset import TauDataset, Jet, pad_collate
from enreg.tools.models.logTrainingProgress import logTrainingProgress_regression
from enreg.tools.models.logTrainingProgress import logTrainingProgress_decaymode

cls_loss = nn.CrossEntropyLoss(reduction="none")
# reg_loss = nn.L1Loss(reduction="none")
reg_loss = nn.HuberLoss(reduction='mean', delta=1.0)
dm_loss = nn.CrossEntropyLoss(reduction="none")


def model_loop(model, tensorboard, idx_epoch, optimizer, ds_loader, dev, is_train=True, kind="ptreg"):
    loss_tot = 0.0
    normalization = 0.0
    pred_vals = []
    true_vals = []
    ratios = []
    for ibatch, batched_jets in enumerate(ds_loader):

        pfs = batched_jets.pfs.to(dev, non_blocking=True)
        pfs_mask = batched_jets.pfs_mask.to(dev, non_blocking=True)
        gen_tau_label = batched_jets.gen_tau_label.to(dev, non_blocking=True)
        true_istau = gen_tau_label!=-1

        pred = model(pfs, pfs_mask)

        if kind == "ptreg":
            #pred = log(ptgen/ptreco) -> ptgen = exp(pred)*ptreco
            assert(pred.shape[1] == 1)
            pred = torch.squeeze(pred, dim=-1)
            gen_tau_pt = batched_jets.gen_tau_pt.to(dev, non_blocking=True)
            reco_jet_pt = batched_jets.reco_jet_pt.to(dev, non_blocking=True)
            target = torch.log(gen_tau_pt/reco_jet_pt)
            pred_pt = torch.exp(pred) * reco_jet_pt
            pred_vals.append(pred_pt[gen_tau_label!=-1].detach().cpu())
            true_vals.append(gen_tau_pt[gen_tau_label!=-1].cpu())
            ratios.extend((pred_pt.detach().cpu()/gen_tau_pt.cpu()).detach().cpu().numpy())
            loss = reg_loss(pred, target)
            normalization += torch.flatten(loss).size(dim=0)
            # loss = torch.sum(loss).item()
            if is_train:
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            loss_tot += loss.detach().cpu().sum().item()


        elif kind == "dm_multiclass":
            assert(pred.shape[1] == 16)
            pred_vals.append(torch.argmax(pred[true_istau], axis=-1).detach().cpu())
            true_vals.append(gen_tau_label[true_istau].cpu())
            target_onehot = torch.nn.functional.one_hot(gen_tau_label[true_istau].long(), 16).float()
            loss = dm_loss(pred[true_istau], target_onehot)
            loss = torch.sum(loss) / torch.sum(true_istau)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_tot += loss.detach().cpu().item()

        else:
            raise Exception("Unknown kind={}".format(kind))

    if kind == "ptreg":
        loss_tot /= normalization
        mean_reco_gen_ratio = np.mean(np.abs(ratios))
        median_reco_gen_ratio = np.median(np.abs(ratios))
        stdev_reco_gen_ratio = np.std(np.abs(ratios))
        iqr_reco_gen_ratio = np.quantile(np.abs(ratios), 0.75) - np.quantile(np.abs(ratios), 0.25)
        logTrainingProgress_regression(
            tensorboard,
            idx_epoch,
            "train" if is_train else "validation",
            loss_tot,
            mean_reco_gen_ratio,
            median_reco_gen_ratio,
            stdev_reco_gen_ratio,
            iqr_reco_gen_ratio,
            None,
        )
    elif kind == "dm_multiclass":
        logTrainingProgress_decaymode(
            tensorboard,
            idx_epoch,
            "train" if is_train else "validation",
            loss_tot,
            None,
        )
    return loss_tot, pred_vals, true_vals


@hydra.main(config_path="../config", config_name="model_training", version_base=None)
def trainSimpleDNN(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_paths = []
    validation_paths = []

    for sample in cfg.samples_to_use:
        print(sample)
        train_paths.extend(cfg.datasets.train[sample])
        validation_paths.extend(cfg.datasets.validation[sample])

    training_data = g.load_all_data(train_paths, n_files=cfg.n_files)
    validation_data = g.load_all_data(validation_paths, n_files=cfg.n_files)
    dataset_train = TauDataset(data=training_data)
    dataset_validation = TauDataset(data=validation_data)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.models.SimpleDNN.training.batch_size,
        num_workers=cfg.models.SimpleDNN.training.num_dataloader_workers,
        prefetch_factor=cfg.models.SimpleDNN.training.prefetch_factor,
        shuffle=True,
        collate_fn=pad_collate
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=cfg.models.SimpleDNN.training.batch_size,
        num_workers=cfg.models.SimpleDNN.training.num_dataloader_workers,
        prefetch_factor=cfg.models.SimpleDNN.training.prefetch_factor,
        shuffle=True,
        collate_fn=pad_collate
    )

    if cfg.models.SimpleDNN.training.type == 'regression':
        kind = "ptreg"
    elif cfg.models.SimpleDNN.training.type == 'dm_multiclass':
        kind = "dm_multiclass"

    if kind == "ptreg":
        model = DeepSet(1).to(device)
    elif kind == "dm_multiclass":
        model = DeepSet(16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # lr_scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=1.0e-3,
    #     epochs=cfg.models.ParticleTransformer.training.num_epochs,
    #     steps_per_epoch=num_batches_train,
    #     anneal_strategy="cos"
    # )

    tensorboard = SummaryWriter(os.path.join(cfg.output_dir, f"tensorboard_{datetime.datetime.now()}"))
    min_loss_validation = -1.0
    for idx_epoch in range(cfg.models.SimpleDNN.training.num_epochs):
        loss_train, _, _ = model_loop(
            model,
            tensorboard,
            idx_epoch,
            optimizer,
            dataloader_train,
            device,
            is_train=True,
            kind=kind
        )
        loss_val, pred_val_reg, true_val_reg = model_loop(
            model,
            tensorboard,
            idx_epoch,
            optimizer,
            dataloader_validation,
            device,
            is_train=False,
            kind=kind
        )
        if min_loss_validation == -1.0 or loss_val < min_loss_validation:
            print("Found new best model :)")
            best_model_file = cfg.models.SimpleDNN.training.model_file.replace(".pt", "_best.pt")
            print("Saving best model to file %s" % best_model_file)
            best_model_output_path = os.path.join(cfg.output_dir, best_model_file)
            torch.save(model.state_dict(), best_model_output_path)
            print("Done.")
            min_loss_validation = loss_val
    print("Finished training")
    model_output_path = os.path.join(cfg.output_dir, cfg.models.SimpleDNN.training.model_file)
    print("Saving model to file %s" % model_output_path)
    torch.save(model.state_dict(), model_output_path)
    print("Done.")
    tensorboard.close()

# TODO: Add OneCycleLR

if __name__ == '__main__':
    trainSimpleDNN()