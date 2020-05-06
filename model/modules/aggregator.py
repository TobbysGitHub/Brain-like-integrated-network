import torch
from torch import nn

from model.modules.components.unit_wise_GRU import UnitWiseGRU
from model.modules.components.unit_wise_linear import UnitWiseLinear


class AggregatorRegion(nn.Module):
    def __init__(self, num_units, dim_unit, dim_inputs, dim_hidden, max_bptt):
        super().__init__()

        self.num_units = num_units
        self.dim_unit = dim_unit
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.dim_outputs = num_units * dim_unit
        self.max_bptt = max_bptt
        self.model = nn.Sequential(
            # nn.LayerNorm(self.dim_inputs),
            UnitWiseLinear(num_units, self.dim_inputs, self.dim_hidden),
            UnitWiseGRU(num_units, self.dim_hidden, self.dim_hidden, self.max_bptt),
            UnitWiseLinear(num_units, self.dim_hidden, self.dim_unit)
        )

    def forward(self, x):
        """
        :param x: s_b * dim_i
        :return:
        """

        x = x.view(-1, 1, self.dim_inputs)
        x = x.expand(-1, self.num_units, -1)
        x = self.model(x)

        return x

    def reset(self):
        pass
        # self.model[2].reset()

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}, ' \
               'max_bptt:{max_bptt}'.format(**self.__dict__)


class Aggregator(nn.Module):
    def __init__(self,
                 num_units_regions,
                 dim_unit,
                 dim_outputs_regions,
                 dim_hidden,
                 max_bptt):
        super().__init__()
        self.num_units_regions = num_units_regions
        self.dim_unit = dim_unit

        self.dim_inputs_regions = [d1 + d2 for d1, d2 in zip(dim_outputs_regions, dim_outputs_regions[1:] + [0])]
        self.dim_hidden = dim_hidden
        self.max_bptt = max_bptt

        self.region_list = nn.ModuleList()

        for num_units, dim_inputs in zip(num_units_regions, self.dim_inputs_regions):
            self.region_list.append(AggregatorRegion(num_units=num_units,
                                                     dim_unit=dim_unit,
                                                     dim_inputs=dim_inputs,
                                                     dim_hidden=dim_hidden,
                                                     max_bptt=max_bptt))

    def forward(self, x0, x1):
        results = []
        x1 = list(x1[1:]) + [None]
        for x_0, x_1, agg in zip(x0, x1, self.region_list):
            x = torch.cat([x_0, x_1], dim=1) if x_1 is not None else x_0
            x = x.detach()
            result = agg(x)
            results.append(result)

        return results

    def reset(self):
        for agg in self.region_list:
            agg.reset()

    def extra_repr(self) -> str:
        return 'num_units_regions:{num_units_regions}, ' \
               'dim_unit:{dim_unit}, ' \
               'dim_inputs_regions:{dim_inputs_regions}, ' \
               'dim_hidden:{dim_hidden}, ' \
               'max_bptt:{max_bptt}'.format(**self.__dict__)


if __name__ == '__main__':
    pass
