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

    # def forward(self, x, int_matrix=None, mask=None, training=False):

    #     # What do B, N and C correspond to.

    #     B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    #     # Project the inputs all at once.
    #     qkv = self.qkv(x)


    #     # Reshape the projected output so that they're segregated in terms of
    #     # query, key, and value projections.
    #     qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))

    #     # Transpose so that the `num_heads` becomes the leading dimensions.
    #     # Helps to better segregate the representation sub-spaces.
    #     qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
    #     scale = tf.cast(self.scale, dtype=qkv.dtype)
    #     q, k, v = qkv[0] * scale, qkv[1], qkv[2]

    #     # Obtain the raw attention scores.
    #     attn = tf.matmul(q, k, transpose_b = True)

    #     # Normalize the attention scores.

    #     if int_matrix is not None:
    #         attn += int_matrix

    #     if mask is not None:
    #         mask = tf.cast(mask, dtype=attn.dtype)
    #         mask = tf.tile(mask, [1, tf.shape(attn)[1], 1, 1])
    #         attn += (1.0 - mask)*-1e9

    #     attn = self.softmax(attn)

    #     # Final set of projections as done in the vanilla attention mechanism.
    #     x = tf.matmul(attn, v)
    #     x = tf.transpose(x, perm=[0, 2, 1, 3])
    #     x = tf.reshape(x, (B, N, C))
        
    #     x = self.proj(x)
    #     x = self.proj_drop(x, training=training)
    #     return x, attn


# class LayerScale(layers.Layer):
#     def __init__(self, init_values, projection_dim, **kwargs):
#         super(LayerScale, self).__init__(**kwargs)
#         self.init_values = init_values
#         self.projection_dim = projection_dim
#         self.gamma_initializer = tf.keras.initializers.Constant(self.init_values)

#     def build(self, input_shape):
#         # Ensure the layer is properly built by defining its weights in the build method
#         self.gamma = self.add_weight(
#             shape=(self.projection_dim,),
#             initializer=self.gamma_initializer,
#             trainable=True,
#             name='gamma'
#         )
#         super(LayerScale, self).build(input_shape)

#     def call(self, inputs,mask=None):
#         # Element-wise multiplication of inputs and gamma
#         if mask is not None:
#             return inputs * self.gamma* mask
#         else:
#             return inputs * self.gamma

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



