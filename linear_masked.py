import torch
import torch.nn as nn


class LinearMasked(nn.Linear):
    """
    Class implementing nn.Linear with mask
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)
