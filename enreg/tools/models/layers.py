import torch
from torch import nn


class StochasticDepth(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Stochastic Depth layer (https://arxiv.org/abs/1603.09382).

    Reference:
        https://github.com/rwightman/pytorch-image-models
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: float, **kwargs):  # num_skip = num_keep ???
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0], x.shape[1], 1)  # Expecting inputs of shape (num_jets, num_features, num_particles)
            random_tensor = keep_prob + torch.rand(shape)
            random_tensor = torch.floor(random_tensor)
            x[:, self.num_skip:, :] = x[:, self.num_skip:, :] * random_tensor
        return x


class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super(LayerScale, self).__init__()
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma = nn.Parameter(torch.full((self.projection_dim,), self.init_values))

    def forward(self, inputs, mask=None):
        # Element-wise multiplication of inputs and gamma
        if mask is not None:
            return inputs * self.gamma * mask
        else:
            return inputs * self.gamma


class SimpleHeadAttention(nn.Module):
    """Simple MHA where masks can be directly added to the inputs.
    Args:
        projection_dim (int): projection dimension for the query, key, and value
            of attention.
        num_heads (int): number of attention heads.
        dropout_rate (float): dropout rate to be used for dropout in the attention
            scores as well as the final projected outputs.
    """
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(projection_dim * 3)
        self.proj = nn.Linear(projection_dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)



