import torch
from torch import nn
import numpy as np

from model import opt_parser
from model.sub_layers.modules.unit_wise_memory import UnitWiseMemory
from model.sub_layers.modules.random_mask import RandomMask
from model.sub_layers.modules.unit_wise_linear import UnitWiseLinear


class AttentionLayer(nn.Module):
    def __init__(self, num_units, opt):
        super().__init__()
        self.opt = opt
        self.num_units = num_units
        self.dim_inputs = opt.dim_attention
        self.dim_outputs = opt.dim_attention_unit
        self.linear = UnitWiseLinear(num_units, self.dim_inputs, self.dim_outputs,
                                     bias=False)

        self.attention_mask = RandomMask(p=opt.attention_mask_p)
        self.memory = UnitWiseMemory(num_units, opt)
        self.temperature = nn.Parameter(torch.ones(num_units) * np.sqrt(self.dim_outputs))

    def forward(self, attention, value):
        assert value is not None
        if attention is None:
            return (None,) * 5

        attention = attention.view(self.opt.batch_size, 1, self.dim_inputs) \
            .expand(self.opt.batch_size, self.num_units, self.dim_inputs)  # s_b * n_u * d_input

        query = self.linear(attention)

        if self.memory.fill(query, value) > 0:
            return (None,) * 5

        weights = self.cal_weights(query)
        rewards, outputs = self.cal_outputs(weights)

        return outputs, self.memory.values, weights, rewards, lambda reward: self.memory.refresh(weights, query, value, reward)

    def cal_weights(self, query):
        # bmm(n_units * n_capacity * d_key, batch_size * n_units * d_key * 1)
        weights = torch.matmul(self.memory.keys, query.unsqueeze(-1))
        weights = weights.squeeze(-1)  # batch_size * n_units * n_capacity
        weights /= self.temperature.unsqueeze(-1)
        weights = self.attention_mask(weights)
        weights = torch.softmax(weights, dim=-1)
        return weights

    def cal_outputs(self, weights):
        with torch.no_grad():
            # batch_size * n_units * n_capacity
            weights = weights * self.memory.rewards
            rewards = weights.sum(-1)

            weights /= rewards.unsqueeze(-1)

            # mul(n_units * n_capacity * d_value, batch_size * n_units * n_capacity * 1)
            outputs = self.memory.values * weights.unsqueeze(-1)

            outputs = outputs.sum(-2)  # batch_size * n_units * d_value
        return rewards, outputs

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)


def main():
    opt = opt_parser.parse_opt()
    n_units = 2
    opt.dim_attention = 32
    opt.dim_attention_unit = 8
    opt.batch_size = 512

    l = AttentionLayer(n_units, opt)

    attention = torch.rand(size=(opt.batch_size, opt.dim_attention))
    x = torch.rand(opt.batch_size, n_units, opt.dim_outputs_unit)
    y = l(attention, x)
    y = l(attention, x)

    reward = torch.zeros(size=(opt.batch_size,))
    y[-1](reward)
    y[-1](reward)
    pass


if __name__ == '__main__':
    main()
