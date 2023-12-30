import copy

import torch.nn as nn


class RepModel(nn.Module):
    def __init__(self):
        super().__init__()

    def reparameterize(self):
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()

    def forward(self, x):
        raise NotImplementedError
