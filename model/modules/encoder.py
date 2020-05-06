import torch
from torch import nn

from model.modules.components.unit_wise_linear import UnitWiseLinear


class EncoderRegion(nn.Module):

    def __init__(self, num_units, dim_inputs, dim_hidden, dim_unit):
        super().__init__()

        self.num_units = num_units
        self.dim_unit = dim_unit
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.dim_outputs = num_units * dim_unit
        self.model = nn.Sequential(
            # nn.LayerNorm(dim_inputs),
            UnitWiseLinear(num_units, dim_inputs, dim_hidden),
            nn.LeakyReLU(),
            UnitWiseLinear(num_units, dim_hidden, dim_unit),
        )

    def forward(self, x):
        """
        :param x: s_b * d_input
        """
        x = x.view(-1, 1, self.dim_inputs) \
            .expand(-1, self.num_units, -1).clone()  # s_b * n_u * d_input

        x = self.model(x)

        return x

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)


class Encoder(nn.Module):
    def __init__(self,
                 num_units_regions,
                 dim_unit,
                 dim_hidden):
        super().__init__()
        self.dim_inputs = 4 * (96 * 96)
        self.num_units_regions = num_units_regions
        self.dim_unit = dim_unit
        self.dim_hidden = dim_hidden

        self.linear = nn.Linear(self.dim_inputs, 256)
        self.region_list = nn.ModuleList()

        for i, num_units in enumerate(self.num_units_regions):
            if i == 0:
                dim_inputs = 256
            else:
                dim_inputs = self.region_list[i - 1].dim_outputs
            self.region_list.append(EncoderRegion(num_units=num_units,
                                                  dim_unit=dim_unit,
                                                  dim_inputs=dim_inputs,
                                                  dim_hidden=dim_hidden))
        self.dim_outputs_regions = [enc.dim_outputs for enc in self.region_list]
        self.dim_outputs = sum(self.dim_outputs_regions)

    def forward(self, inputs):

        inputs = inputs.detach()
        x = self.linear(inputs)

        result = []
        for enc in self.region_list:
            x = enc(x)
            result.append(x)
            x = x.detach()

        return result

    def extra_repr(self) -> str:
        return 'num_units_regions:{num_units_regions}, ' \
               'dim_outputs_regions:{dim_outputs_regions}' \
               'dim_hidden:{dim_hidden}, ' \
               'dim_unit:{dim_unit}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)


if __name__ == '__main__':
    pass
