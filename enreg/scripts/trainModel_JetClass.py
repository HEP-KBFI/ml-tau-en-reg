#!/usr/bin/python3
from comet_ml import Experiment
import os
import json
import hydra
import datetime
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
from torch.utils.tensorboard import SummaryWriter

from enreg.tools import general as g
from enreg.tools.losses.FocalLoss import FocalLoss
from enreg.tools.models.Lookahead import Lookahead
from enreg.tools.losses.initWeights import initWeights

from enreg.tools.models.ParticleTransformer import ParticleTransformer

from enreg.tools.data_management.jetclass_data_manager import IterableJetClassDataset

# from enreg.tools.data_management.particleTransformer_dataset import load_row_groups, ParticleTransformerDataset
# See asendada enda loaderiga

from enreg.tools.models.logTrainingProgress import logTrainingProgress_decaymode


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


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
    num_classes,
    train=True,
    use_comet=True,
    experiment=None,
):
    if train:
        print("::::: TRAIN LOOP :::::")
    else:
        print("::::: VALID LOOP :::::")

    loss_train = 0.0
    normalization = 0.0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    if train:
        print("Setting model to train mode")
        model.train()
        tensorboard_tag = "train"
    else:
        print("Setting model to eval mode")
        model.eval()
        tensorboard_tag = "valid"
    print(len(dataloader_train))

    for idx_batch, (cand_features, cand_kinematics, mask, y) in tqdm.tqdm(
        enumerate(dataloader_train), total=len(dataloader_train)
    ):
        # Compute prediction and loss

        pred = model(
            cand_features.to(device=dev),
            cand_kinematics.to(device=dev),
            mask.to(device=dev),
        ).to(device=dev)

        loss = loss_fn(pred, y.to(device=dev))
        loss_train += loss.sum().item()
        normalization += torch.flatten(loss).size(dim=0)

        pred_dm = torch.argmax(pred.detach().cpu(), axis=-1).numpy()
        true_dm = torch.argmax(y.cpu(), axis=-1).numpy()
        confusion_matrix += sklearn.metrics.confusion_matrix(
            true_dm, pred_dm, labels=range(num_classes)
        )

        # Backpropagation
        if train:
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # lr_scheduler.step()

    loss_train /= normalization
    if use_comet:
        if train:
            experiment.log_metric(name="loss_train", value=loss_train, step=idx_epoch)
            experiment.log_confusion_matrix(
                y_true=None,
                y_predicted=None,
                matrix=None,
                labels=None,
                title="Confusion Matrix",
                row_label="Actual Category",
                column_label="Predicted Category",
                step=idx_epoch,
            )

        else:
            experiment.log_metric(
                name="loss_validation", value=loss_train, step=idx_epoch
            )

    logging_data = logTrainingProgress_decaymode(
        tensorboard,
        idx_epoch,
        tensorboard_tag,
        loss_train,
        np.zeros(5),  # TODO?
        confusion_matrix,
    )
    tensorboard.flush()
    print("Loss = {}".format(loss_train))
    return loss_train, logging_data


@hydra.main(config_path="../config", config_name="jetclass", version_base=None)
def trainModel(cfg: DictConfig) -> None:
    print("<trainModel>:")
    experiment = None
    try:
        experiment = Experiment()
        if cfg.comet.experiment is None:
            now = datetime.datetime.now()
            experiment_name = now.strftime("%d/%m/%Y, %H:%M:%S")
        else:
            experiment_name = cfg.comet.experiment
        experiment.set_name(experiment_name)
        use_comet = True
        print(f"Using CometML for logging. Experiment name {cfg.comet.experiment}")
    except ValueError:
        use_comet = False
        print("CometML API key not found, not logging to CometML")

    # because we are doing plots for tensorboard, we don't want anything to crash
    plt.switch_backend("agg")

    model_output_path = os.path.join(cfg.training.output_dir, "model_out")

    # if we are going to train the model, ensure a model does not already exist (e.g. if doing testing as a separate step)
    if cfg.train and os.path.isdir(model_output_path):
        raise Exception(
            "Output directory exists while train=True: {}".format(model_output_path)
        )

    dataset_train = IterableJetClassDataset(
        data_dir=cfg.jetclass_parquet_dir, dataset_type="train", cfg=cfg
    )
    dataset_validation = IterableJetClassDataset(
        data_dir=cfg.jetclass_parquet_dir, dataset_type="val", cfg=cfg
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    print("Building model...")

    # configure the number of model input dimensions based on the input features
    input_dim = 13

    num_classes = len(cfg.labels)
    model = ParticleTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        num_layers=cfg.models.ParticleTransformer.hyperparameters.num_layers,
        embed_dims=cfg.models.ParticleTransformer.hyperparameters.embed_dims,
        use_pre_activation_pair=False,
        for_inference=False,
        use_amp=False,
        metric="eta-phi",
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

        loss_fn = nn.CrossEntropyLoss(reduction="none")

        base_optimizer = torch.optim.RAdam(model.parameters(), lr=cfg.training.lr)

        if cfg.training.slow_optimizer == "Lookahead":
            print(
                "Using {} optimizer with Lookahead.".format(cfg.training.fast_optimizer)
            )
            optimizer = Lookahead(base_optimizer=base_optimizer, k=10, alpha=0.5)
        elif cfg.training.slow_optimizer == "None":
            print("Using {} optimizer.".format(cfg.training.fast_optimizer))
            optimizer = base_optimizer
        else:
            raise RuntimeError("Invalid configuration parameter 'slow_optimizer' !!")

        num_batches_train = len(dataloader_train)
        print("Training for {} epochs.".format(cfg.training.trainer.max_epochs))
        print("#batches(train) = {}".format(num_batches_train))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer,
            T_max=len(dataloader_train) * cfg.training.trainer.max_epochs,
            eta_min=cfg.training.lr * 0.01,
        )
        print("Starting training...")
        print(" current time:", datetime.datetime.now())
        tensorboard = SummaryWriter(os.path.join(model_output_path, "tensorboard_logs"))
        min_loss_validation = -1.0
        # early_stopper = EarlyStopper(patience=10)
        for idx_epoch in range(cfg.training.trainer.max_epochs):
            print("Processing epoch #%i" % idx_epoch)
            print(" current time:", datetime.datetime.now())

            train_loss, train_logging_data = train_loop(
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
                num_classes,
                use_comet=use_comet,
                experiment=experiment,
            )
            # print("lr = {}".format(lr_scheduler.get_last_lr()[0]))
            # tensorboard.add_scalar("lr", lr_scheduler.get_last_lr()[0], idx_epoch)

            with torch.no_grad():
                loss_validation, val_logging_data = train_loop(
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
                    num_classes,
                    train=False,
                    use_comet=use_comet,
                    experiment=experiment,
                )

            if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
                print("Saving best model to file {}".format(best_model_output_path))
                torch.save(model.state_dict(), best_model_output_path)
                min_loss_validation = loss_validation

            new_history_data = {}
            for key, value in val_logging_data.items():
                new_history_data[key + "_validation"] = value
            for key, value in train_logging_data.items():
                new_history_data[key + "_train"] = value

            history_data_path = os.path.join(model_output_path, "history.json")
            if os.path.exists(history_data_path):
                with open(history_data_path, "rt") as in_file:
                    history_data = json.load(in_file)
                for key, value in new_history_data.items():
                    history_data[key].append(value)
            else:
                history_data = {}
                for key, value in new_history_data.items():
                    history_data[key] = []
                    history_data[key].append(value)

            with open(history_data_path, "wt") as out_file:
                json.dump(history_data, out_file, indent=4, cls=NumpyEncoder)

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
            dataset_full = IterableJetClassDataset(
                data_dir=cfg.jetclass_parquet_dir, dataset_type="test", cfg=cfg
            )

            # test dataloader must NOT specify num_workers or prefetch,
            # otherwise the order of jets in the dataset can change
            # and thus make subsequent evaluation incorrect
            dataloader_full = DataLoader(
                dataset_full,
                batch_size=cfg.training.batch_size,
            )
            preds = []
            targets = []
            for idx_batch, (cand_features, cand_kinematics, mask, y) in tqdm.tqdm(
                dataloader_full, total=len(dataloader_full)
            ):
                with torch.no_grad():
                    pred = model(cand_features, cand_kinematics, mask)
                    pred = torch.softmax(pred, axis=-1)
                    pred = torch.argmax(pred, axis=-1)
                preds.extend(pred.detach().cpu().numpy())
                targets.extend(y.detach().cpu().numpy())
            preds = np.array(preds)
            targets = np.array(targets)
            data_to_save = {
                "pred": preds,
                "target": targets,
            }
            ak.to_parquet(
                ak.Record({kind: data_to_save}),
                os.path.join(model_output_path, test_sample),
            )  # TODO


if __name__ == "__main__":
    trainModel()
