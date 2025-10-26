import torch
import torch.nn as nn
from gabbro.models.backbone_base import BackboneModel
from enreg.tools.models.ParticleTransformer import ParticleTransformer
from gabbro.models.vqvae import VQVAELightning
from omegaconf import OmegaConf, DictConfig
from pathlib import Path


class OmniParT(ParticleTransformer):
    def __init__(
            self,
            input_dim: int,
            cfg: DictConfig,
            num_classes=None,
            # network configurations
            pair_input_dim: int = 4,
            pair_extra_dim: int = 0,
            remove_self_pair: bool = False,
            use_pre_activation_pair: bool = True,
            embed_dims: list[int] = [256, 512, 256],
            pair_embed_dims: list[int] = [64, 64, 64],
            num_heads: int = 8,
            num_layers: int = 8,
            num_cls_layers: int = 2,
            block_params=None,
            cls_block_params: dict = {"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
            fc_params: list = [],
            activation: str = "gelu",
            # misc
            trim : bool = True,
            for_inference: bool = False,
            use_amp: bool = False,
            metric: str = "eta-phi",
            verbosity: int = 0,
            **kwargs
    ):
        super().__init__(input_dim=input_dim,
                         num_classes=num_classes,
                         # network configurations
                         pair_input_dim=pair_input_dim,
                         pair_extra_dim=pair_extra_dim,
                         remove_self_pair=remove_self_pair,
                         use_pre_activation_pair=use_pre_activation_pair,
                         embed_dims=embed_dims,
                         pair_embed_dims=pair_embed_dims,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         num_cls_layers=num_cls_layers,
                         block_params=block_params,
                         cls_block_params=cls_block_params,
                         fc_params=fc_params,
                         activation=activation,
                         # misc
                         trim=trim,
                         for_inference=for_inference,
                         use_amp=use_amp,
                         metric=metric,
                         verbosity=verbosity,
                         **kwargs
                         )
        self.cfg = cfg
        self.embed = EmbedParT(self.cfg)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.frozen_parameters = False
        if self.cfg.version == "from_scratch":
            # Train head & BB
            print("Training OmniParT from scratch.")
        if self.cfg.version == "fixed_backbone":
            # Train only head, do not train the pre-trained BB
            print("Training only OmniParT head. Backbone parameters are frozen.")
        if self.cfg.version == "fine_tuning":
            # Train head, fine tune BB after epoch 30.
            print("Training OmniParT head. Backbone parameters are unfrozen after initial epochs.")

    def forward(self, cand_features, cand_kinematics_pxpypze=None, cand_mask=None, frost='freeze'):
        # cand_features: (N=num_batches, C=num_features, P=num_particles)
        # cand_kinematics_pxpypze: (N, 4, P) [px,py,pz,energy]
        # cand_mask: (N, 1, P) -- real particle = 1, padded = 0
        padding_mask = ~cand_mask.squeeze(1)  # (N, 1, P) -> (N, P)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            num_particles = cand_features.size(-1)
            if self.cfg.version == "from_scratch":
                # Train head & BB
                for param in self.embed.bb_model.parameters():
                    param.requires_grad = True
            if self.cfg.version == "fixed_backbone":
                # Train only head, do not train the pre-trained BB
                for param in self.embed.bb_model.parameters():
                    param.requires_grad = False
            if self.cfg.version == "fine_tuning":
                # Train head, fine tune BB after epoch 30.
                if frost == 'freeze' and not self.frozen_parameters:
                    print("Freezing parameters")
                    for param in self.embed.bb_model.parameters():
                        param.requires_grad = False
                    self.frozen_parameters = True
                elif frost == 'unfreeze' and self.frozen_parameters:
                    print("Unfreezing parameters")
                    for param in self.embed.bb_model.parameters():
                        param.requires_grad = True
                    self.frozen_parameters = False


            # OmniJet-alpha embedding (VQ-VAE + BB)
            cand_features_embed = self.embed(cand_features, cand_mask).permute(1, 0, 2)

            # Transformer part. In contrast to ParT we don't use particle attention here.
            for block in self.blocks:
                cand_features_embed = block(cand_features_embed, x_cls=None, padding_mask=padding_mask)

            # transform per-jet class tokens
            cls_tokens = self.cls_token.expand(1, cand_features_embed.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(cand_features_embed, x_cls=cls_tokens, padding_mask=padding_mask)
            x_cls = self.norm(cls_tokens).squeeze(0)

            output = self.fc(x_cls)  # (N, num_class)

            return output


class EmbedParT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vqvae_model = VQVAELightning.load_from_checkpoint(cfg.ckpt_path).to(device='cpu')
        self.vqvae_model.eval()
        # Freeze VQ-VAE parameters
        for param in self.vqvae_model.parameters():
            param.requires_grad = False
        ckpt_cfg = OmegaConf.load(Path(cfg.ckpt_path).parent / "config.yaml")
        self.pp_dict = OmegaConf.to_container(ckpt_cfg.data.dataset_kwargs_common.feature_dict)
        n_GPT_blocks = 1
        print(f"Using {n_GPT_blocks} GPT blocks")
        self.bb_model = BackboneModel(
            embedding_dim=256,
            attention_dropout=0.0,
            vocab_size=32002,
            max_sequence_len=128,
            n_heads=32,
            n_GPT_blocks=n_GPT_blocks
        )
        if self.cfg.version != "from_scratch":
            loaded_bb_model = torch.load(cfg.bb_path, map_location=torch.device('cpu'))
            gpt_state = {k.replace("module.", ""): v for k, v in loaded_bb_model["state_dict"].items() if
                         k.startswith("module.")}
            self.bb_model.load_state_dict(gpt_state, strict=False)


    def forward(self, cand_omni_kinematics, cand_mask):
        # preprocess according to self.pp_dict
        cand_omni_kinematics[:, 0] = torch.nan_to_num(
            torch.log(cand_omni_kinematics[:, 0]) - self.pp_dict['part_pt']['subtract_by'] * self.pp_dict['part_pt'][
                'multiply_by'],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        # As the next preprocessing parts would just cut out some particles, then for the time being these are not used.
        # (cand_omni_kinematics[:, 1] > pp_dict['part_etarel']['larger_than']) & (cand_omni_kinematics[:, 1] < pp_dict['part_etarel']['smaller_than'])
        # (cand_omni_kinematics[:, 1] > pp_dict['part_phirel']['larger_than']) & (cand_omni_kinematics[:, 1] < pp_dict['part_phirel']['smaller_than'])
        x_particle_reco, vq_out = self.vqvae_model.forward(
            cand_omni_kinematics.permute(0, 2, 1),  # To be in accordance with the order expected by the function
            torch.squeeze(cand_mask)  # Get rid of the axis=1
        )
        encoded_jets = self.bb_model(torch.squeeze(vq_out['q'], axis=2), padding_mask=torch.squeeze(cand_mask))
        return encoded_jets
