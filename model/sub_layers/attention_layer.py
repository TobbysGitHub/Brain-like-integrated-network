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

        self.memory = UnitWiseMemory(num_units, opt)
        self.attention_mask = RandomMask(p=opt.attention_mask_p)
        self.temperature = nn.Parameter(torch.ones(num_units) * np.sqrt(self.dim_outputs))

    def forward(self, attention, value):
        assert value is not None
        if attention is None:
            return (None,) * 3

        attention = attention.view(self.opt.batch_size, 1, self.dim_inputs) \
            .expand(self.opt.batch_size, self.num_units, self.dim_inputs)  # s_b * n_u * d_input

        query = self.linear(attention)

        if self.memory.fill(query, value) > 0:
            return (None,) * 3

        weights = self.cal_weights(query, self.memory.keys)
        outputs = self.cal_outputs(weights, self.memory.values, self.memory.rewards)

        return weights, outputs, lambda reward: self.memory.refresh(weights, query, value, reward)

    def cal_weights(self, query, keys):
        # bmm(n_units * n_capacity * d_key, batch_size * n_units * d_key * 1)
        weights = torch.matmul(keys, query.unsqueeze(-1))
        weights = weights.squeeze(-1)  # batch_size * n_units * n_capacity
        weights /= self.temperature.unsqueeze(-1)
        weights = self.attention_mask(weights)
        weights = torch.softmax(weights, dim=-1)
        return weights

    @staticmethod
    def cal_outputs(weights, values, rewards):
        with torch.no_grad():
            # batch_size * n_units * n_capacity
            weights = weights * rewards

            weights /= weights.sum(-1, keepdim=True)

            # mul(n_units * n_capacity * d_value, batch_size * n_units * n_capacity * 1)
            outputs = values * weights.unsqueeze(-1)

            outputs = outputs.sum(-2)  # batch_size * n_units * d_value
        return weights, outputs


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
