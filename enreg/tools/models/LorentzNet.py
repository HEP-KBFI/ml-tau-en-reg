import torch
from torch import nn
from typing import Tuple


class LGEB(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 0,
        dropout: float = 0.0,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ) -> None:
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # dims for Minkowski norm & inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        if not last_layer:
            self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

        self.last_layer = last_layer

    def m_model(self, hi: torch.Tensor, hj: torch.Tensor, norms: torch.Tensor, dots: torch.Tensor) -> torch.Tensor:
        out = torch.cat([hi, hj, norms, dots], dim=-1)
        out = out.view(-1, out.size(dim=-1))
        out = self.phi_e(out)
        out = out.reshape(self.batchsize, -1, out.size(dim=-1))
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h: torch.Tensor, segment_ids: torch.Tensor, m: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        agg = unsorted_segment_sum(m, segment_ids, num_segments=self.n_particles)
        agg = torch.cat([h, agg, node_attr], dim=-1)
        agg = agg.view(-1, agg.size(dim=-1))
        agg = self.phi_h(agg)
        agg = agg.reshape(self.batchsize, -1, agg.size(dim=-1))
        out = h + agg
        return out

    def x_model(self, x: torch.Tensor, segment_ids: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "phi_x")
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100.0, max=+100.0)
        agg = unsorted_segment_mean(trans, segment_ids, num_segments=self.n_particles)
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(
        self, edgei: torch.Tensor, edgej: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edgei = edgei.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xi = torch.gather(x, 1, edgei)
        edgej = edgej.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xj = torch.gather(x, 1, edgej)
        x_diff = xi - xj
        norms = normsq4(x_diff).unsqueeze(dim=-1)
        dots = dotsq4(xi, xj).unsqueeze(dim=-1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, edgei: torch.Tensor, edgej: torch.Tensor, node_attr: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.batchsize = h.size(dim=0)
        assert x.size(dim=0) == self.batchsize
        assert edgei.size(dim=0) == self.batchsize
        assert edgej.size(dim=0) == self.batchsize
        segment_ids = edgei

        self.n_particles = h.size(dim=-2)

        norms, dots, x_diff = self.minkowski_feats(edgei, edgej, x)

        hi = torch.gather(h, 1, edgei.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        hj = torch.gather(h, 1, edgej.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        m = self.m_model(hi, hj, norms, dots)  # [B*N, hidden]

        if not self.last_layer:
            x = self.x_model(x, segment_ids, x_diff, m)
        h = self.h_model(h, segment_ids, m, node_attr)
        return h, x, m


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    # CV: the following einsum operator gives the same as the operation
    #       'result = segment_ids @ data'
    #     but makes it more explicit how to multiply tensors in three dimensions
    result = torch.einsum("ijk,ikl->ijl", segment_ids, data)
    return result


def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`."""
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    # CV: the following einsum operators give the same as the operations
    #       'result = segment_ids @ data' and 'count = segment_ids @ torch.ones_like(data)',
    #     but make it more explicit how to multiply tensors in three dimensions
    result = torch.einsum("ijk,ikl->ijl", segment_ids, data)
    count = torch.einsum("ijk,ikl->ijl", segment_ids, torch.ones_like(data))
    result = result / count.clamp(min=1)
    return result


def normsq4(p: torch.Tensor) -> torch.Tensor:
    r"""Minkowski square norm
    |p|^2 = p[0]^2 - p[1]^2 - p[2]^2 - p[3]^2
    """
    psq = torch.pow(p, 2)
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""Minkowski inner product
    <p,q> = p[0]q[0] - p[1]q[1] - p[2]q[2] - p[3]q[3]
    """
    psq = p * q
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result


def psi(p: torch.Tensor) -> torch.Tensor:
    """psi(p) = Sgn(p) * log(|p| + 1)"""
    result = torch.sign(p) * torch.log(torch.abs(p) + 1)
    return result


class LorentzNet(nn.Module):
    r"""Implementation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    """

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        n_class: int = 2,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        verbosity: int = 0,
    ) -> None:
        if verbosity >= 2:
            print("<LorentzNet::LorentzNet>:")
            print(" n_scalar = %i" % n_scalar)
            print(" n_hidden = %i" % n_hidden)
            print(" n_class = %i" % n_class)
            print(" n_layers = %i" % n_layers)
            print(" c_weight = %1.3f" % c_weight)
            print(" dropout = %1.2f" % dropout)
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    self.n_hidden,
                    self.n_hidden,
                    self.n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.n_hidden, n_class)
        )  # classification

        self.verbosity = verbosity

    def forward(self, x: torch.Tensor, scalars: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:

        #embed the per-particle non Lorentz invariant quantities (scalars)
        h = self.embedding(scalars)

        #create particle-to-particle "edges" within each jet with all-to-all connections
        n_particles = x.size(dim=1)
        edges = torch.ones(n_particles, n_particles, dtype=torch.long, device=h.device)
        edges_above_diag = torch.triu(edges, diagonal=1)
        edges_below_diag = torch.tril(edges, diagonal=-1)
        edges = torch.add(edges_above_diag, edges_below_diag)
        edges = torch.nonzero(edges)
        edges = torch.swapaxes(edges, 0, 1)
        edgei = torch.unsqueeze(edges[0], dim=0)
        edgei = edgei.expand(h.size(dim=0), -1)
        edgej = torch.unsqueeze(edges[1], dim=0)
        edgej = edgej.expand(h.size(dim=0), -1)

        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i].forward(h, x, edgei, edgej, node_attr=scalars)
        h = h * node_mask
        h = h.view(-1, n_particles, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        result = pred.squeeze(0)
        return result