import torch
from torch import nn

from model import opt_parser
from model.sub_layers.modules.unit_wise_linear import UnitWiseLinear


class EncodeLayer(nn.Module):

    def __init__(self, num_units, dim_inputs, opt):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs = dim_inputs
        self.dim_hidden = opt.dim_hid_enc_unit
        self.dim_outputs = opt.dim_outputs_unit
        self.opt = opt
        self.model = nn.Sequential(
            nn.LayerNorm(dim_inputs),
            UnitWiseLinear(num_units, dim_inputs, opt.dim_hid_enc_unit),
            nn.LeakyReLU(),
            UnitWiseLinear(num_units, opt.dim_hid_enc_unit, opt.dim_outputs_unit),
        )

    def forward(self, x):
        """
        :param x: s_b * d_input
        """
        x = x.view(self.opt.batch_size, 1, self.dim_inputs) \
            .expand(self.opt.batch_size, self.num_units, self.dim_inputs)  # s_b * n_u * d_input

        x = self.model(x)

        return x

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, dim_inputs:{dim_inputs}, ' \
               'dim_hidden:{dim_hidden}, dim_outputs:{dim_outputs}'.format(**self.__dict__)


if __name__ == '__main__':
    opt = opt_parser.parse_opt()
    opt.batch_size = 2
    opt.dim_hid_enc_unit = 6
    opt.dim_outputs_unit = 8

    num_units = 3
    dim_inputs = 4

    e = EncodeLayer(num_units, dim_inputs, opt)

    x = torch.rand(size=(opt.batch_size, dim_inputs))
    y = e(x)
