import torch
from torch import nn


class UnitWiseLinear(nn.Module):

    def __init__(self, n_units, in_features, out_features, bias=True, init=True):
        super().__init__()
        self.n_units = n_units
        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.rand(n_units, out_features, in_features))
        self.bias = bias
        self.init = init
        if bias:
            self.b = nn.Parameter(torch.zeros(n_units, out_features))

        if init:
            nn.init.xavier_normal_(self.w)

    def forward(self, x):
        """
        :param x: s_b * n_u * d_input
        :return: s_b * n_u * d_output
        """
        x = torch.matmul(self.w, x.unsqueeze(-1)).squeeze(-1)

        if self.bias:
            x += self.b

        return x

    def extra_repr(self) -> str:
        return 'n_units={n_units}, in_features={in_features}, ' \
               'out_features={out_features}, bias={bias}'.format(**self.__dict__)


if __name__ == '__main__':
    batch_size = 4
    n_units = 2
    d_inputs = 8
    d_outputs = 6

    x = torch.ones(size=(batch_size, n_units, d_inputs))
    l = UnitWiseLinear(n_units, d_inputs, d_outputs)

    y = l(x)
