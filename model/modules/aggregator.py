import torch
from torch import nn

from model.modules.components.unit_wise_GRU import UnitWiseGRU
from model.modules.components.unit_wise_linear import UnitWiseLinear


class Aggregator(nn.Module):
    def __init__(self, num_units, dim_unit, dim_inputs1, dim_inputs2, dim_hidden, max_bptt):
        super().__init__()

        self.num_units = num_units
        self.dim_unit = dim_unit
        self.dim_inputs1 = dim_inputs1
        self.dim_inputs2 = dim_inputs2
        self.dim_inputs = dim_inputs1 + dim_inputs2
        self.dim_hidden = dim_hidden
        self.dim_outputs = num_units * dim_unit
        self.max_bptt = max_bptt
        self.model = nn.Sequential(
            nn.LayerNorm(self.dim_inputs),
            UnitWiseLinear(num_units, self.dim_inputs, self.dim_hidden),
            UnitWiseGRU(num_units, self.dim_hidden, self.dim_hidden, self.max_bptt),
            UnitWiseLinear(num_units, self.dim_hidden, self.dim_unit)
        )

    def forward(self, x1, x2):
        """
        :param x1: s_b * dim_i1
        :param x2: s_b * dim_i2
        :return:
        """
        if x2 is None:
            if self.dim_inputs2 == 0:
                x2 = torch.zeros(0, device=x1.device)
            else:
                return None

        x1 = x1.view(x1.shape[0], 1, self.dim_inputs1)
        x2 = x2.view(x1.shape[0], 1, self.dim_inputs2)

        x = torch.cat([x1, x2], dim=-1) \
            .expand(-1, self.num_units, -1)

        x = self.model(x)

        return x

    def reset(self):
        self.model[2].reset()

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_unit:{dim_unit}, ' \
               'dim_inputs:({dim_inputs1}+{dim_inputs2}={dim_inputs}), ' \
               'dim_hidden:{dim_hidden}, ' \
               'dim_outputs:{dim_outputs}, ' \
               'max_bptt:{max_bptt}'.format(**self.__dict__)


if __name__ == '__main__':
    pass
