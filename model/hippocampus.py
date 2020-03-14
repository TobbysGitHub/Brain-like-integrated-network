import numpy as np
import torch
from torch import nn

from model.modules.components.random_mask import RandomMask


class Hippocampus(nn.Module):
    def __init__(self, num_units_regions, dim_inputs, dim_attention_global, dim_attention_unit, num_attention_groups, mask_p):
        super().__init__()
        self.num_units = sum(num_units_regions)
        self.dim_inputs = dim_inputs
        self.dim_outputs = self.num_units * dim_attention_unit
        self.dim_attention_global = dim_attention_global
        self.dim_attention_unit = dim_attention_unit
        self.num_groups = num_attention_groups
        self.mask_p = mask_p

        self.model = nn.Sequential(
            nn.Linear(in_features=self.dim_inputs, out_features=dim_attention_global),
            nn.Linear(in_features=dim_attention_global, out_features=self.num_units * self.dim_attention_unit)
        )

        def random_mask_fn(x: torch.Tensor):
            if self.mask_p <= 0:
                return x
            mask = torch.rand_like(x, device=x.device) < self.mask_p
            return x.masked_fill_(mask, -np.inf)

        def eye_mask_fn(x: torch.Tensor):
            mask = torch.eye(x.shape[0], device=x.device) == 1
            return x.masked_fill_(mask.unsqueeze(-1), -np.inf)

        self.random_mask = random_mask_fn
        self.eye_mask = eye_mask_fn
        self.temperature = nn.Parameter(torch.ones(self.num_units))

        # self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x, memories=None, eye_mask=True):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.dim_inputs)
        # project to units
        attention = self.model(x)
        attention = attention.view(-1, self.num_units, self.dim_attention_unit)

        if memories is None:
            return attention
        else:
            # memories is a tuple
            mem_attention = memories[0]  # s_b * num_units(mem_capacity)
            mem_value = memories[1]  # s_b * num_units(mem_capacity) * d

            weights = (attention.unsqueeze(1) * mem_attention).sum(-1)  # s_b * s_b * num_units
            weights = weights * self.temperature

            weights = self.random_mask(weights)
            if eye_mask:
                weights = self.eye_mask(weights)

            weights = weights.view(batch_size, batch_size // self.num_groups, self.num_groups, self.num_units)
            weights = torch.softmax(weights, dim=1)
            weights = weights.view(batch_size, batch_size, self.num_units) / 8

            outputs = torch.sum(mem_value * weights.unsqueeze(-1), dim=1)  # s_b * num_units * d
            return attention, weights, outputs

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}, ' \
               'dim_attention_global:{dim_attention_global}, ' \
               'dim_attention_unit:{dim_attention_unit}, ' \
               'mask_p:{mask_p}'.format(**self.__dict__)


def main():
    pass


if __name__ == '__main__':
    main()
