import torch
import tqdm ### why
from enreg.tools import general as g
from torch.utils.data import DataLoader
from enreg.tools.data_management.simpleDNN_dataset import TauDataset
from torch.utils.tensorboard import SummaryWriter


cls_loss = nn.CrossEntropyLoss(reduction="none")
reg_loss = nn.L1Loss(reduction="none")
dm_loss = nn.CrossEntropyLoss(reduction="none")

#given multiple jets with a variable number of PF candidates per jet, create 3d-padded arrays
#in the shape [Njets, Npfs_max, Nfeat]
def pad_collate(jets):
    pfs = [jet.pfs for jet in jets]
    pfs_mask = [jet.pfs_mask for jet in jets]
    gen_tau_label = [jet.gen_tau_label for jet in jets]
    gen_tau_pt = [jet.gen_tau_pt for jet in jets]
    reco_jet_pt = [jet.reco_jet_pt for jet in jets]
    pfs = torch.nn.utils.rnn.pad_sequence(pfs, batch_first=True)
    pfs_mask = torch.nn.utils.rnn.pad_sequence(pfs_mask, batch_first=True)
    gen_tau_label = torch.concatenate(gen_tau_label, axis=0)
    gen_tau_pt = torch.concatenate(gen_tau_pt, axis=0)
    reco_jet_pt = torch.concatenate(reco_jet_pt, axis=0)
    return Jet(
        pfs=pfs,
        pfs_mask=pfs_mask,
        reco_jet_pt=reco_jet_pt,
        gen_tau_label=gen_tau_label,
        gen_tau_pt=gen_tau_pt
    )


def model_loop(model, optimizer, ds_loader, dev, lr_scheduler, is_train=True, kind="ptreg"):
    loss_tot = 0.0
    pred_vals = []
    true_vals = []
    for ibatch, batched_jets in enumerate(tqdm.tqdm(ds_loader, total=len(ds_loader), ncols=80)):

        pfs = batched_jets.pfs.to(dev, non_blocking=True)
        pfs_mask = batched_jets.pfs_mask.to(dev, non_blocking=True)
        gen_tau_label = batched_jets.gen_tau_label.to(dev, non_blocking=True)
        true_istau = gen_tau_label!=-1

        pred = model(pfs, pfs_mask)

        if kind == "binary":
            assert(pred.shape[1] == 2)
            pred_vals.append(pred[:, 1].detach().cpu())
            true_vals.append(true_istau.cpu())
            loss = cls_loss(pred, true_istau.long()).mean()
        elif kind == "ptreg":
            #pred = log(ptgen/ptreco) -> ptgen = exp(pred)*ptreco
            assert(pred.shape[1] == 1)
            pred = torch.squeeze(pred, dim=-1)
            gen_tau_pt = batched_jets.gen_tau_pt.to(dev, non_blocking=True)
            reco_jet_pt = batched_jets.reco_jet_pt.to(dev, non_blocking=True)

            target = torch.log(gen_tau_pt/reco_jet_pt)
            
            pred_pt = torch.exp(pred) * reco_jet_pt
            pred_vals.append(pred_pt[gen_tau_label!=-1].detach().cpu())
            true_vals.append(gen_tau_pt[gen_tau_label!=-1].cpu())
            loss = reg_loss(pred[true_istau], target[true_istau])
            loss = torch.sum(loss) / torch.sum(true_istau)
        elif kind == "dm_multiclass":
            assert(pred.shape[1] == 16)
            pred_vals.append(torch.argmax(pred[true_istau], axis=-1).detach().cpu())
            true_vals.append(gen_tau_label[true_istau].cpu())
            target_onehot = torch.nn.functional.one_hot(gen_tau_label[true_istau].long(), 16).float()
            loss = dm_loss(pred[true_istau], target_onehot)
            loss = torch.sum(loss) / torch.sum(true_istau)
        else:
            raise Exception("Unknown kind={}".format(kind))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step() ##
        loss_tot += loss.detach().cpu().item()
    return loss_tot, pred_vals, true_vals


@hydra.main(config_path="../config", config_name="model_training", version_base=None)
def trainSimpleDNN(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model = DeepSet(1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=1.0e-3,
        epochs=cfg.models.ParticleTransformer.training.num_epochs,
        steps_per_epoch=num_batches_train,
        anneal_strategy="cos"
    )

    tensorboard = SummaryWriter(os.path.join(cfg.output_dir, f"tensorboard_{datetime.datetime.now()}"))

    for idx_epoch in range(cfg.models.SimpleDNN.training.num_epochs):
        loss_train, _, _ = model_loop(model_ptreg, optimizer, dl_sig_train, device, is_train=True, kind="ptreg")
        loss_val, pred_val_reg, true_val_reg = model_loop(model_ptreg, optimizer, dl_sig_val, device, is_train=False, kind="ptreg")
        print("{} L={:.2f}/{:.2f}".format(iepoch, loss_train, loss_val))
        losses_train.append(loss_train)
        losses_val.append(loss_val)

losses_train = []
losses_val = []

for iepoch in range(20):