import numpy as np
import torch
from torch import nn

from model.sub_layers.modules.random_mask import RandomMask


class Memory(nn.Module):
    def __init__(self, num_units_layer, mask_p):
        super().__init__()
        self.num_units_layer = num_units_layer
        self.mask_p = mask_p
        self.attentions = None  # num_units * capacity * d_att_u
        self.memories = None  # num_units * capacity * d
        self.mask = RandomMask(mask_p)
        self.tmpr = nn.Parameter(torch.ones(self.num_units, 1) * np.sqrt(self.dim_attention_unit))

    def load(self, attentions, memories):
        self.attentions = attentions
        self.memories = memories

    def forward(self, attention):
        # attention: s_b * num_units * d_att_u
        weights = torch.matmul(self.attentions, attention.unsqueeze(-1)).squeeze(-1)  # s_b * num_units * capacity
        weights = self.mask(weights)

        weights /= self.tmpr
        weights = torch.softmax(weights, dim=-1)

        outputs = torch.sum(self.memories * weights.unsqueeze(-1), dim=-2)  # s_b * num_units * d

        return outputs, weights, self.memories
