import torch
from torch import nn

from model import opt_parser
from model.sub_layers.modules.unit_wise_GRU import UnitWiseGRU
from model.sub_layers.modules.unit_wise_linear import UnitWiseLinear


class AggregateLayer(nn.Module):
    def __init__(self, num_units, dim_input1, dim_input2, opt):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs1 = dim_input1
        self.dim_inputs2 = dim_input2
        self.dim_inputs = dim_input1 + dim_input2
        self.dim_hidden = opt.dim_hid_agg_unit
        self.dim_outputs = opt.dim_outputs_unit
        self.max_bptt = opt.max_bptt_agg
        self.opt = opt
        self.model = nn.Sequential(
            nn.LayerNorm(self.dim_inputs),
            UnitWiseLinear(num_units, self.dim_inputs, self.dim_hidden),
            UnitWiseGRU(num_units, self.dim_hidden, self.dim_hidden, self.max_bptt),
            UnitWiseLinear(num_units, self.dim_hidden, self.dim_outputs)
        )

    def forward(self, x1, x2):
        """
        :param x1: s_b * dim_i1
        :param x2: s_b * dim_i2
        :return:
        """
        if x1 is None or x2 is None:
            return None

        x1 = x1.view(self.opt.batch_size, 1, self.dim_inputs1)
        x2 = x2.view(self.opt.batch_size, 1, self.dim_inputs2)

        x = torch.cat([x1, x2], dim=-1) \
            .expand(self.opt.batch_size, self.num_units, self.dim_inputs)

        x = self.model(x)

        return x

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, dim_inputs:({dim_inputs1}+{dim_inputs2}={dim_inputs}), ' \
               'dim_hidden:{dim_hidden}, dim_outputs:{dim_outputs}, max_bptt:{max_bptt}'.format(**self.__dict__)


if __name__ == '__main__':
    opt = opt_parser.parse_opt()
    opt.batch_size = 2
    opt.dim_hid_agg_unit = 6
    opt.dim_outputs_unit = 8

    opt.max_bptt_agg = 10

    num_units = 3
    dim_inputs1 = 4
    dim_inputs2 = 5

    e = AggregateLayer(num_units, dim_inputs1, dim_inputs2, opt)

    x1 = torch.rand(size=(opt.batch_size, dim_inputs1))
    x2 = torch.rand(size=(opt.batch_size, dim_inputs2))

    for i in range(10):
        y = e(x1, x2)
