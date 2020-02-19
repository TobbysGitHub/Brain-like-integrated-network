import torch
from torch import nn
import numpy as np

from model import opt_parser
from model.sub_layers.modules.unit_wise_linear import UnitWiseLinear


class AttentionLayer(nn.Module):
    def __init__(self, num_units, opt):
        super().__init__()
        self.opt = opt
        self.num_units = num_units
        self.dim_inputs = opt.dim_attention
        self.dim_outputs = opt.dim_attention_unit
        self.model = UnitWiseLinear(num_units, self.dim_inputs, self.dim_outputs,
                                    bias=False)

        self.temperature = nn.Parameter(torch.ones(num_units) * np.sqrt(self.dim_outputs))

    def forward(self, x):
        x = x.view(self.opt.batch_size, 1, self.dim_inputs) \
            .expand(self.opt.batch_size, self.num_units, self.dim_inputs)  # s_b * n_u * d_input

        x = self.model(x)

        return x


def main():
    opt = opt_parser.parse_opt()
    n_units = 2
    opt.dim_attention = 32
    opt.dim_attention_unit = 8
    opt.batch_size = 3

    l = AttentionLayer(n_units, opt)

    x = torch.rand(size=(opt.batch_size, opt.dim_attention))
    y = l(x)
    assert y.shape == (opt.batch_size, n_units, opt.dim_attention_unit)
    pass


if __name__ == '__main__':
    main()
