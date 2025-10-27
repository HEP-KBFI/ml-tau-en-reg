import math

from torch import nn


def initWeights(model):
    for module in model.modules():
        if type(module) in [nn.Embedding]:
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif type(module) in [nn.Linear, nn.Conv1d, nn.Conv1d, nn.Conv3d]:
            nn.init.kaiming_normal_(
                module.weight.data, a=0, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                    module.weight.data
                )
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(module.bias, -bound, +bound)
