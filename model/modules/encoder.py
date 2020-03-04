import torch
from torch import nn

from model.modules.components.unit_wise_linear import UnitWiseLinear


class Encoder(nn.Module):

    def __init__(self, num_units, dim_inputs, dim_hidden, dim_unit):
        super().__init__()

        self.num_units = num_units
        self.dim_unit = dim_unit
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.dim_outputs = num_units * dim_unit
        self.model = nn.Sequential(
            nn.LayerNorm(dim_inputs),
            UnitWiseLinear(num_units, dim_inputs, dim_hidden),
            nn.LeakyReLU(),
            UnitWiseLinear(num_units, dim_hidden, dim_unit),
        )

    def forward(self, x):
        """
        :param x: s_b * d_input
        """
        x = x.view(-1, 1, self.dim_inputs) \
            .expand(-1, self.num_units, -1)  # s_b * n_u * d_input

        x = self.model(x)

        return x

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_unit:{dim_unit}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_hidden:{dim_hidden}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)


if __name__ == '__main__':
    pass
