import torch
import torch.nn as nn




class PET(nn.Module):
    """Point Edge Transformer """
    def __init__(
        self,
        num_feat,
        num_jet,
        num_classes=2,
        num_keep=7,  # Number of features that wont be dropped
        feature_drop=0.1,
        projection_dim=128,
        local=True,
        K=10,
        num_local=2,
        num_layers=8,
        num_class_layers=2,
        num_gen_layers=2,
        num_heads=4,
        drop_probability=0.0,
        simple=False,
        layer_scale=True,
        layer_scale_init=1e-5,
        talking_head=False,
        mode='classifier',
        num_diffusion=3,
        dropout=0.0,
        class_activation=None,
     ):
        super().__init()
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init=layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.ema = 0.999
        self.class_activation = class_activation

        input_features = torch.zeros(1, num_feat)
        input_points = torch.zeros(1, 2)
        input_mask = torch.zeros(1, 1)
        input_jet = torch.zeros(num_jet)
        input_label = torch.zeros(num_classes)
        input_time = torch.zeros(1)
