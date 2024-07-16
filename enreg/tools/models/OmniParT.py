import torch
import torch.nn as nn
from gabbro.models.gpt_model import BackboneModel
from enreg.tools.models.ParticleTransformer import ParticleTransformer
from gabbro.models.vqvae import VQVAELightning

class OmniParT(ParticleTransformer):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        metric="eta-phi",
        verbosity=0,
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
        self.embed = EmbedParT()
        self.for_inference = for_inference
        self.use_amp = use_amp


    def forward(self, cand_features, tau_data_transf, cand_kinematics_pxpypze=None, cand_mask=None):
        # cand_features: (N=num_batches, C=num_features, P=num_particles)
        # cand_kinematics_pxpypze: (N, 4, P) [px,py,pz,energy]
        # cand_mask: (N, 1, P) -- real particle = 1, padded = 0
        padding_mask = ~cand_mask.squeeze(1) # (N, 1, P) -> (N, P)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            num_particles = cand_features.size(-1)

            parT_features = self.embed(tau_data_transf, cand_mask)
            cand_features_embed = parT_features

            attn_mask = None
            if cand_kinematics_pxpypze is not None and self.pair_embed is not None:
                attn_mask = self.pair_embed(cand_kinematics_pxpypze).view(-1, num_particles, num_particles) # (N*num_heads, P, P)

            # transform particles
            for block in self.blocks:
                cand_features_embed = block(cand_features_embed, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
            
            # transform per-jet class tokens
            cls_tokens = self.cls_token.expand(1, cand_features_embed.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(cand_features_embed, x_cls=cls_tokens, padding_mask=padding_mask)
            x_cls = self.norm(cls_tokens).squeeze(0)

            output = self.fc(x_cls) #(N, num_class)

            return output


class EmbedParT(nn.Module):
    def __init__(self):
        super().__init__()
        ckpt_path = "/home/laurits/ml-tau-en-reg/enreg/omnijet_alpha/checkpoints/vqvae_8192_tokens/model_ckpt.ckpt"
        # ckpt_path = cfg.ckpt_path

        bb_path = "/home/laurits/ml-tau-en-reg/enreg/omnijet_alpha/checkpoints/generative_8192_tokens/OmniJet_generative_model_UnintentionalPinscher_59.ckpt"
        # bb_path = cfg.bb_path

        self.vqvae_model = VQVAELightning.load_from_checkpoint(ckpt_path).to(device)
        self.vqvae_model.eval()

        ckpt_cfg = OmegaConf.load(Path(ckpt_path).parent / "config.yaml")
        self.pp_dict = OmegaConf.to_container(ckpt_cfg.data.dataset_kwargs_common.feature_dict)

        loaded_bb_model = torch.load(bb_path, map_location=torch.device('cpu'))
        self.bb_model = BackboneModel(256, 0.0, 8194, 128, 32, 3)  # Reason for the magic numbers (e.g. 8192+2)
        gpt_state = {k.replace("module.", ""): v for k, v in loaded_bb_model["state_dict"].items() if k.startswith("module.")}
        self.bb_model.load_state_dict(gpt_state)

    def forward(self, cand_omni_kinematics, cand_mask):
        
        # preprocess according to self.pp_dict
        cand_omni_kinematics[:, 0] = torch.nan_to_num(
            torch.log(cand_omni_kinematics[:, 0]) - self.pp_dict['part_pt']['subtract_by'] * self.pp_dict['part_pt']['multiply_by'],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        # As the next preprocessing parts would just cut out some particles, then for the time being these are not used.
        # (cand_omni_kinematics[:, 1] > pp_dict['part_etarel']['larger_than']) & (cand_omni_kinematics[:, 1] < pp_dict['part_etarel']['smaller_than'])
        # (cand_omni_kinematics[:, 1] > pp_dict['part_phirel']['larger_than']) & (cand_omni_kinematics[:, 1] < pp_dict['part_phirel']['smaller_than'])
        x_particle_reco, vq_out  = self.vqvae_model.forward(
            cand_omni_kinematics.permute(0, 2, 1), # To be in accordance with the order expected by the function
            torch.squeeze(cand_mask) # Get rid of the axis=1 
        )
        encoded_jets = self.bb_model(torch.squeeze(vq_out['q'], axis=2), padding_mask=torch.squeeze(cand_mask))
        return encoded_jets
