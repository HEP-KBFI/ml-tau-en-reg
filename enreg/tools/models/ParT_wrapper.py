import torch
import lightning as L
from omegaconf import DictConfig
from enreg.tools.models.Lookahead import Lookahead
from enreg.tools.models.ParticleTransformer import ParticleTransformer
import sklearn.metrics as sklm

# Lookahead optimizer with k=6 and alpha=0.5 to minimize cross-entropy loss
# Inner optimizer is Radam with beta1=0.95 and beta2=0.999 and epsilon=10^-5
# Batch size 512
# Initial lr = 0.001
# No weight decay
# Train 1M iterations corresponding to ~5 epochs over the full training set.
# LR remains constant for the first 70% of iterations, then decays exponentially at an interval of every 20k iterations
#   down to 1% of the inital value at the end of the training
# They use checkpoint with highest accuracy.

class ParTModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_classes: int):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.cfg = cfg
        self.ParT = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=cfg.models.ParticleTransformer.hyperparameters.num_layers,
            embed_dims=cfg.models.ParticleTransformer.hyperparameters.embed_dims,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric='eta-phi',
        )

    def training_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        loss = self.loss_fn(predicted_labels, target).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)  # Aga need siin one-hot-encoded
        loss = self.loss_fn(predicted_labels, target).mean()
        # background_rejection_rate = "foobar"
        # auc = sklm.roc_auc_score(target, predicted_labels, multi_class="ovo", average="macro")
        # accuracy = sklm.accuracy_score()
        # f1_score = sklm.f1_score()
        self.log("val_loss", loss)
        # self.log("val_rejX", background_rejection_rate)
        # self.log("val_auc", float(auc))
        return loss

    def configure_optimizers(self):
        base_optimizer = torch.optim.RAdam(params=self.ParT.parameters(), lr=self.cfg.training.lr, betas=(0.95, 0.999))
        # optimizer = Lookahead(base_optimizer=base_optimizer, k=6, alpha=0.5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer,
            T_max=20000 * self.cfg.training.trainer.max_epochs,
            # T_max=len(dataloader_train) * cfg.training.num_epochs,
            eta_min=self.cfg.training.lr * 0.01
        )
        # LR remains constant for the first 70% of iterations, then decays exponentially at an
        # interval of every 20k iterations down to 1% of the inital value at the end of the training
        return [base_optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)[0]

    def test_step(self, batch, batch_idx):
        return self.forward(batch)[0]

    def forward(self, batch):
        cand_features, cand_kinematics_pxpypze, cand_mask, target = batch
        predicted_labels = self.ParT(
            cand_features=cand_features,
            cand_kinematics_pxpypze=cand_kinematics_pxpypze,
            cand_mask=cand_mask
        )
        return predicted_labels, target