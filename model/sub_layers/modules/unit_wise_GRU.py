import numpy as np
import torch
from torch import nn

from model.sub_layers.modules.unit_wise_linear import UnitWiseLinear


class UnitWiseGRU(nn.Module):
    def __init__(self, n_units, input_size, hidden_size, max_bptt=np.inf):
        super().__init__()
        self.n_units = n_units
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.max_bptt = max_bptt
        self.counter = 0

        self.ir = UnitWiseLinear(n_units, input_size, hidden_size, init=False)
        self.hr = UnitWiseLinear(n_units, hidden_size, hidden_size, init=False)
        self.iz = UnitWiseLinear(n_units, input_size, hidden_size, init=False)
        self.hz = UnitWiseLinear(n_units, hidden_size, hidden_size, init=False)
        self.in_ = UnitWiseLinear(n_units, input_size, hidden_size, init=False)
        self.hn = UnitWiseLinear(n_units, hidden_size, hidden_size, init=False)

        std = np.sqrt(1 / hidden_size)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=std)

    def forward(self, x):
        """
        :param x: s_b * n_units * d_input
        """
        if self.counter == 0:
            self.hidden = torch.zeros(x.shape[0], x.shape[1], self.hidden_size,
                                      device=x.device)

        r = torch.sigmoid(self.ir(x) + self.hr(self.hidden))
        z = torch.sigmoid(self.iz(x) + self.hz(self.hidden))
        n = torch.tanh(self.in_(x) + r * self.hn(self.hidden))

        x = (1 - z) * n + z * self.hidden

        self.counter += 1
        if self.counter % self.max_bptt == 0:
            self.hidden = x.detach()
        else:
            self.hidden = x

        return x

    def extra_repr(self) -> str:
        return 'n_units={n_units}, input_size={input_size}, ' \
               'hidden_size={hidden_size}, max_bptt={max_bptt}'.format(**self.__dict__)


if __name__ == '__main__':
    batch_size = 4
    n_units = 2
    d_inputs = 8
    d_outputs = 6

    x1 = torch.ones(size=(batch_size, n_units, d_inputs))
    x2 = torch.ones(size=(batch_size, n_units, d_inputs))

    lg = UnitWiseGRU(n_units, d_inputs, d_outputs)
    y1 = lg(x1)
    y2 = lg(x2)
