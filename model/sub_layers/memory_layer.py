import torch
from torch import nn

from model import opt_parser
from model.sub_layers import AttentionLayer
from model.sub_layers.modules.random_mask import RandomMask
from model.sub_layers.modules.state_module import StatefulModule


class MemoryLayer(nn.Module):
    # noinspection PyArgumentList
    def __init__(self, n_units, opt):
        super().__init__()
        self.n_units = n_units
        self.capacity = opt.memory_capacity
        self.d_key, self.d_value = opt.dim_attention_unit, opt.dim_outputs_unit

        self.fresh_counter = 0
        # self.max_bptt = opt.max_bptt_mem
        self.mem_refresh_rate = lambda: opt.memory_fresh_rate
        self.reward_gamma = lambda: opt.reward_gamma
        self.reward_refresh_rate = lambda: opt.reward_fresh_rate

        self.opt = opt

        self.attention_net = AttentionLayer(n_units, opt)
        self.attention_mask = RandomMask(p=opt.attention_mask_p)

        self.num_empty = nn.Parameter(torch.tensor([self.capacity]),
                                      requires_grad=False)
        self.key_mem = nn.Parameter(torch.empty(n_units, self.capacity, self.d_key),
                                    requires_grad=False)
        self.value_mem = nn.Parameter(torch.empty(self.n_units, self.capacity, self.d_value),
                                      requires_grad=False)
        self.reward_mem = nn.Parameter(torch.ones(n_units, self.capacity),
                                       requires_grad=False)
        self.reward_weights = None

    def forward(self, attention, value):
        if attention is None or value is None:
            return (None,) * 4

        query = self.attention_net(attention)

        query = query.view(self.opt.batch_size, self.n_units, self.d_key)
        value = value.view(self.opt.batch_size, self.n_units, self.d_value)

        # fill memory in the first few steps
        if self._fill_empty(key_batch=query, value_batch=value):
            return (None,) * 4

        weights, outputs = self._attention(query)

        # a closure called after cal loss and reward
        def refresh_handler(reward):
            self._refresh_memory(weights, query, value)
            self._refresh_reward(weights, reward)

        return self.value_mem, weights, outputs, refresh_handler

    def _fill_empty(self, key_batch, value_batch):
        num_empty = self.num_empty.item()
        if num_empty == 0:
            return False

        num = min(num_empty, self.opt.batch_size)

        # n_units * batch_size * d__
        key_batch, value_batch = (x[:num].permute(1, 0, 2).detach()
                                  for x in (key_batch, value_batch))

        begin = self.capacity - num_empty

        self.key_mem[:, begin:begin + num] = key_batch
        self.value_mem[:, begin:begin + num] = value_batch
        self.num_empty -= num
        return True

    def _attention(self, query):
        # bmm(n_units * n_capacity * d_key, batch_size * n_units * d_key * 1)
        weights = torch.matmul(self.key_mem, query.unsqueeze(-1))
        weights = weights.squeeze(-1)  # batch_size * n_units * n_capacity
        weights /= self.attention_net.temperature.unsqueeze(-1)
        weights = self.attention_mask(weights)
        weights = torch.softmax(weights, dim=-1)

        with torch.no_grad():
            # batch_size * n_units * n_capacity
            weights = weights * self.reward_mem

            weights /= weights.sum(-1, keepdim=True)

            # mul(n_units * n_capacity * d_value, batch_size * n_units * n_capacity * 1)
            outputs = self.value_mem * weights.unsqueeze(-1)

            outputs = outputs.sum(-2)  # batch_size * n_units * d_value
        return weights, outputs

    def _refresh_memory(self, weights, key_new, value_new):
        """
        :param weights: batch_size * n_units * capacity
        :param key_new: batch_size * n_units * d_key
        :param value_new: batch_size * n_units * d_value
        """
        with torch.no_grad():
            fresh_ratio = weights * self.mem_refresh_rate()  # batch_size * n_units * capacity
            retain_ratio = 1 - fresh_ratio.sum(0)  # n_units * capacity

            # n_units * n_capacity * d_key
            self.key_mem *= retain_ratio.unsqueeze(-1)
            self.key_mem += (fresh_ratio.unsqueeze(-1) * key_new.unsqueeze(-2)).sum(0)
            self.value_mem *= retain_ratio.unsqueeze(-1)
            self.value_mem += (fresh_ratio.unsqueeze(-1) * value_new.unsqueeze(-2)).sum(0)
        self.fresh_counter += 1

        # if self.fresh_counter % self.max_bptt == 0:
        #     self.value_mem.detach_()

    def _refresh_reward(self, weights, reward):
        with torch.no_grad():
            # batch_size * n_units * n_capacity
            if self.reward_weights is None:
                self.reward_weights = weights
            else:
                self.reward_weights = self.reward_weights * self.reward_gamma() + weights
            fresh_ratio = self.reward_weights * self.reward_refresh_rate()
            retain_ratio = 1 - fresh_ratio.sum(0)
            # n_units * n_capacity
            self.reward_mem *= retain_ratio
            self.reward_mem += (fresh_ratio * reward.view(-1, 1, 1)).sum(0)


def main():
    opt = opt_parser.parse_opt()
    n_units = 2
    opt.dim_attention = 32
    opt.dim_attention_unit = 8
    opt.dim_outputs_unit = 9
    opt.batch_size = 3
    opt.memory_capacity = 3

    l = MemoryLayer(n_units, opt)

    attention = torch.rand(size=(opt.batch_size, opt.dim_attention))
    value = torch.rand(size=(opt.batch_size, n_units, opt.dim_outputs_unit))

    y = l(attention, value)
    y = l(attention, value)

    reward = torch.zeros(size=(opt.batch_size,))
    y[-1](reward)

    state = l.state_dict()

    l.load_state_dict(state)


if __name__ == '__main__':
    main()
