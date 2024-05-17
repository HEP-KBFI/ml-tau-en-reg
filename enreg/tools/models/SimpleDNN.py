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
    def __init__(self, num_inputs, num_outputs):
        super(DeepSet, self).__init__()

        self.act = nn.GELU
        self.act_obj = self.act()
        self.dropout = 0.2
        self.width = 128
        self.embedding_dim = 128

        # number of inputs
        self.num_pf_features = num_inputs

        self.nn_pf_embedding = ffn(self.num_pf_features, self.embedding_dim, self.width, self.act, self.dropout)
        self.nn_pred = ffn(self.embedding_dim, num_outputs, self.width, self.act, self.dropout)

    def forward(self, pfs_pad, pfs_mask):
        pf_encoded = self.act_obj(self.nn_pf_embedding(pfs_pad))*pfs_mask
        num_pfs = torch.sum(pfs_mask, axis=1)
        jet_encoded1 = self.act_obj(torch.sum(pf_encoded, axis=1)/num_pfs)
        ret = self.nn_pred(jet_encoded1)
        return ret
