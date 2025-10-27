from torch import optim
import lightning as L


def get_target(training_type):
    pass


class TauRegresser(L.LightningModule):
    """ For tau energy regression """
    def __init__(self, model, model_type, training_type,):
        super().__init__()
        self.model = model
        self.training_type = training_type
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def training_step(self, batch, batch_step):
        X, y, weight = batch
        model_inputs = unpack_data(X, self.dev, feature_set, self.model_type)
        # if cfg.model_type == 'OmniParT':
            # if idx_epoch < cfg.models.OmniParT.num_rounds_frozen_backbone:
            #     frost = 'freeze'
            # else:
            #     frost = 'unfreeze'
        pred = model(*model_inputs).to(device=dev)[:, 0]
        y = y["jet_regression"]
        loss = loss_fn(pred, y)
        if use_per_jet_weights:
            loss = loss * weight

        self.log("train_loss", loss)
        return loss


    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y, weight = batch
        model_inputs = unpack_data(X, self.dev, feature_set, self.model_type)
        pred = model(*model_inputs).to(device=dev)[:, 0]
        y = y["jet_regression"]
        loss = loss_fn(pred, y)
        if use_per_jet_weights:
            loss = loss * weight
        self.log("test_loss", loss)


    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y, weight = batch
        model_inputs = unpack_data(X, self.dev, feature_set, self.model_type)
        pred = model(*model_inputs).to(device=dev)[:, 0]
        y = y["jet_regression"]
        loss = loss_fn(pred, y)
        if use_per_jet_weights:
            loss = loss * weight
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



class TauTagger(L.LightningModule):
    def __init__(self, model, model_type, base_optimizer="AdamW", slow_optimizer="Lookahead"):
        """ For tau binary classifier """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.optimizer = base_optimizer
        # self.dev = 

    def training_step(self, batch, batch_step):
        X, y, weight = batch
        model_inputs = unpack_data(X, self.dev, feature_set, self.model_type)
        if cfg.model_type == 'OmniParT':
            if idx_epoch < cfg.models.OmniParT.num_rounds_frozen_backbone:
                frost = 'freeze'
            else:
                frost = 'unfreeze'
            model_inputs = model_inputs + (frost,)
        pred = model(*model_inputs).to(device=dev)
        y = y["binary_classification"]
        loss = loss_fn(pred, y)
        if use_per_jet_weights:
            loss = loss * weight

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class TauClassifier(L.LightningModule):
    def __init__(self, model, model_type, kind):
        """ For decay mode classification """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.dev = 


    def training_step(self, batch, batch_step):
        X, y, weight = batch
        model_inputs = unpack_data(X, self.dev, feature_set, self.model_type)
        if cfg.model_type == 'OmniParT':
            if idx_epoch < cfg.models.OmniParT.num_rounds_frozen_backbone:
                frost = 'freeze'
            else:
                frost = 'unfreeze'
            model_inputs = model_inputs + (frost,)
        pred = model(*model_inputs).to(device=dev)
        y = y["dm_multiclass"]
        y = torch.nn.functional.one_hot(y, num_classes).float()
        loss = loss_fn(pred, y)
        if use_per_jet_weights:
            loss = loss * weight

        self.log("train_loss", loss)
        return loss








# class TauReconstructor(L.LightningModule):


experiment = Experiment(
    project_name="ml-tau",
    workspace="laurits7"
)

experiment.set_name('Test123')