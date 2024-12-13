import os
import hydra
import lightning as L
from omegaconf import DictConfig
from enreg.tools.models.ParT_wrapper import ParTModule
from enreg.tools.data_management import jetclass_data_manager as jdm
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint


@hydra.main(config_path="../config", config_name="jetclass", version_base=None)
def train(cfg: DictConfig):
    parT = ParTModule(cfg=cfg, input_dim=13, num_classes=10)  # 13 cand_features, 4 cand_kinematics
    models_dir = os.path.join(cfg.training.output_dir, "models")
    log_dir = os.path.join(cfg.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        save_weights_only=True,
        filename="ParT-{epoch:02d}-{val_loss:.2f}"
    )
    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=1000),
            checkpoint_callback,
        ],
        logger=CSVLogger(log_dir, name="peakFinding")
    )
    datamodule = jdm.JetClassDataModule(cfg=cfg)
    trainer.fit(model=parT, datamodule=datamodule)

if __name__ == "__main__":
    train()
