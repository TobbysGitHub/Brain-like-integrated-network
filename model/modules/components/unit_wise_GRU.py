import numpy as np
import torch
from torch import nn

from model.modules.components.unit_wise_linear import UnitWiseLinear


class UnitWiseGRU(nn.Module):
    def __init__(self, num_units, in_features, out_features, max_bptt=np.inf):
        super().__init__()
        self.num_units = num_units
        self.in_features = in_features
        self.out_features = out_features

        self.max_bptt = max_bptt
        self.counter = 0

        self.ir = UnitWiseLinear(num_units, in_features, out_features, init=False)
        self.iz = UnitWiseLinear(num_units, in_features, out_features, init=False)
        self.in_ = UnitWiseLinear(num_units, in_features, out_features, init=False)
        self.hr = UnitWiseLinear(num_units, out_features, out_features, init=False)
        self.hz = UnitWiseLinear(num_units, out_features, out_features, init=False)
        self.hn = UnitWiseLinear(num_units, out_features, out_features, init=False)

        self.hidden_init = nn.Parameter(torch.empty(1, num_units, self.out_features))
        self.hidden = None

        bound = np.sqrt(1 / out_features)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x):
        """
        :param x: s_b * num_units * d_input
        """
        if self.hidden is None:
            self.hidden = self.hidden_init.expand(x.shape[0], -1, -1).clone()

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

    def reset(self):
        self.hidden = None

    def extra_repr(self) -> str:
        return 'num_units={num_units}, in_features={in_features}, ' \
               'out_features={out_features}, max_bptt={max_bptt}'.format(**self.__dict__)


if __name__ == '__main__':
    pass
