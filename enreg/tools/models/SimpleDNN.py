import torch
from torch import nn


def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class DeepSet(nn.Module):
    def __init__(self, num_outputs):
        super(DeepSet, self).__init__()

        self.act = nn.GELU
        self.act_obj = self.act()
        self.dropout = 0.2
        self.width = 128
        self.embedding_dim = 128

        # number of inputs
        self.num_pf_features = 13
        self.num_jet_features = 4

        self.nn_pf_embedding = ffn(self.num_pf_features, self.embedding_dim, self.width, self.act, self.dropout)
        self.nn_jet_embedding = ffn(self.num_jet_features, self.embedding_dim, self.width, self.act, self.dropout)
        self.nn_pred = ffn(2*self.embedding_dim, num_outputs, self.width, self.act, self.dropout)

    def forward(self, pfs_pad, pfs_mask, jets):
        # print("pfs", pfs_pad.shape)
        # print("pfs_mask", pfs_mask.shape)
        # print("jets", jets.shape)

        pfs_mask = torch.unsqueeze(pfs_mask, axis=-1)
        pf_encoded = self.act_obj(self.nn_pf_embedding(pfs_pad))*pfs_mask
        num_pfs = torch.sum(pfs_mask, axis=1)
        jet_encoded1 = self.act_obj(torch.sum(pf_encoded, axis=1)/num_pfs)
        jet_encoded2 = self.act_obj(self.nn_jet_embedding(jets))

        ret = self.nn_pred(torch.concatenate([jet_encoded1, jet_encoded2], axis=-1))
        return ret
