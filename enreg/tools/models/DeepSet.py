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
        self.dropout = 0.1
        self.width = 128
        self.embedding_dim = 128

        # number of inputs
        self.num_pf_features = num_inputs

        self.nn_pf_embedding = ffn(
            self.num_pf_features, self.embedding_dim, self.width, self.act, self.dropout
        )
        self.nn_pred = ffn(
            self.embedding_dim, num_outputs, self.width, self.act, self.dropout
        )

    def forward(self, cand_features, cand_kinematics, cand_mask):
        # cand_features: (N=num_batches, C=num_features, P=num_particles)
        # cand_kinematics_pxpypze: (N, 4, P) [px,py,pz,energy]
        # cand_mask: (N, 1, P) -- real particle = 1, padded = 0
        cand_kinematics = torch.swapaxes(
            cand_kinematics, 1, 2
        )  # (N, 4, P) -> (N, P, 4)
        cand_features = torch.swapaxes(cand_features, 1, 2)  # (N, C, P) -> (N, P, C)
        cand_mask = torch.swapaxes(cand_mask, 1, 2)  # (N, 1, P) -> (N, P, 1)

        pf_encoded = self.act_obj(self.nn_pf_embedding(cand_features)) * cand_mask
        num_pfs = torch.sum(cand_mask, axis=1)
        jet_encoded1 = self.act_obj(torch.sum(pf_encoded, axis=1) / num_pfs)
        ret = self.nn_pred(jet_encoded1)
        return ret
