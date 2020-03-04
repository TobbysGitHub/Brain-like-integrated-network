import numpy as np
import torch
from torch import nn

from model.modules.components.random_mask import RandomMask


class Memory(nn.Module):
    def __init__(self, num_units_regions, mask_p, dim_unit, memory_size, delay):
        super().__init__()
        self.num_units = sum(num_units_regions)
        self.mask_p = mask_p
        self.dim_unit = dim_unit

        self.mask = RandomMask(mask_p)
        self.temperature = nn.Parameter(torch.ones(self.num_units))

        self.memory_size = memory_size
        self.memory_delay = delay
        self._loader = self.memory_loader()
        self.loaded = False
        self.attention = None  # capacity * num_units * d_att_u
        self.memory = None  # capacity * num_units * d

    def load(self, attention, memory):
        return self._loader(attention, memory)

    def forward(self, attention):
        # attention: s_b * num_units * d_att_u
        assert self.loaded and attention is not None
        weights = (attention.unsqueeze(1) * self.attention).sum(-1)  # s_b * capacity * num_units
        weights = self.mask(weights)

        weights /= self.temperature
        weights = torch.softmax(weights, dim=1)

        outputs = torch.sum(self.memory * weights.unsqueeze(-1), dim=1)  # s_b * num_units * d

        return outputs, weights, self.memory

    def memory_loader(self):
        attention_list = []
        memory_list = []
        loadings = 0

        def load(attention, memory):
            # s_b * n_u * d_
            nonlocal attention_list, memory_list, loadings
            if attention is None or memory is None:
                return 0

            attention_list.append(attention)
            memory_list.append(memory)
            loadings += len(attention)

            if loadings >= self.memory_size + self.memory_delay:
                self.attention = torch.cat(attention_list[0:self.memory_size], dim=0)
                self.memory = torch.cat(memory_list[0:self.memory_size], dim=0)
                self.loaded = True

                r = attention_list.pop(0)
                memory_list.pop(0)
                loadings -= len(r)

                return len(self.attention)
            return 0

        return load
