import torch
import torch.nn as nn
from gabbro.models.gpt_model import BackboneModel
from enreg.tools.models.ParticleTransformer import ParticleTransformer
from gabbro.models.vqvae import VQVAELightning
from enreg.tools.models.OmniParT import EmbedParT

class OmniFeedforward(nn.Module):
    def __init__(
        self,
        input_dim,
        cfg,
        num_classes,
        use_amp=False,
):
        super().__init__()
        self.cfg = cfg
        self.use_amp = use_amp
        self.frozen_parameters = False

        #Omnijet backbone
        self.embed = EmbedParT(self.cfg)

        #Feedforward output
        fcs = []
        in_dim = 256
        fcs.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fcs)
    
    def forward(self, cand_features, cand_kinematics_pxpypze=None, cand_mask=None, frost='freeze'):
        padding_mask = ~cand_mask.squeeze(1) # (N, 1, P) -> (N, P)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            num_particles = cand_features.size(-1)

            if frost == 'freeze' and not self.frozen_parameters:
                print("Freezing parameters")
                for param in self.embed.parameters():
                    param.requires_grad = False
                self.frozen_parameters = True
            elif frost == 'unfreeze' and self.frozen_parameters:
                print("Unfreezing parameters")
                for param in self.embed.parameters():
                    param.requires_grad = True
                self.frozen_parameters = False
            parT_features = self.embed(cand_features, cand_mask)
            num_pfs = torch.sum(cand_mask, axis=-1)
            jet_encoded = torch.sum(parT_features, axis=1)/num_pfs
            output = self.fc(jet_encoded)
            return output
