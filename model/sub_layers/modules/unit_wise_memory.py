import torch
from torch import nn

from model import opt_parser
from model.sub_layers.modules.random_mask import RandomMask


class UnitWiseMemory(nn.Module):
    # noinspection PyArgumentList
    def __init__(self, n_units, opt):
        super().__init__()
        self.n_units = n_units
        self.capacity = opt.memory_capacity
        self.d_key, self.d_value = opt.dim_attention_unit, opt.dim_outputs_unit

        self.mem_refresh_rate = lambda: opt.memory_fresh_rate
        self.reward_gamma = lambda: opt.reward_gamma
        self.reward_refresh_rate = lambda: opt.reward_fresh_rate

        self.num_empty = nn.Parameter(torch.tensor([self.capacity]),
                                      requires_grad=False)
        self.keys = nn.Parameter(torch.empty(n_units, self.capacity, self.d_key),
                                 requires_grad=False)
        self.values = nn.Parameter(torch.empty(self.n_units, self.capacity, self.d_value),
                                   requires_grad=False)
        self.rewards = nn.Parameter(torch.ones(n_units, self.capacity),
                                    requires_grad=False)
        self.reward_decay_weights = None

    def fill(self, batch_keys, batch_values):
        num_empty = self.num_empty.item()
        if num_empty == 0:
            return 0

        num = min(num_empty, batch_keys.shape[0])
        begin = self.capacity - num_empty
        with torch.no_grad():
            # n_units * batch_size * d__
            keys_batch, values_batch = (x[:num].permute(1, 0, 2)
                                        for x in (batch_keys, batch_values))
            self.keys[:, begin:begin + num] = keys_batch
            self.values[:, begin:begin + num] = values_batch
        self.num_empty -= num
        return num

    def refresh(self, weights, key, value, reward):
        def refresh_memory(weights, key_new, value_new):
            """
            :param weights: batch_size * n_units * capacity
            :param key_new: batch_size * n_units * d_key
            :param value_new: batch_size * n_units * d_value
            """
            with torch.no_grad():
                fresh_ratio = weights * self.mem_refresh_rate()  # batch_size * n_units * capacity
                fresh_ratio = fresh_ratio.unsqueeze(-1)
                retain_ratio = 1 - fresh_ratio.sum(0)  # n_units * capacity * 1

                self.keys *= retain_ratio
                # n_units * n_capacity * d_key
                self.keys += (fresh_ratio * key_new.unsqueeze(-2)).sum(0)
                self.values *= retain_ratio
                self.values += (fresh_ratio * value_new.unsqueeze(-2)).sum(0)

        def refresh_reward(weights, reward):
            """
            :param weights:
            :param reward: batch_size
            """
            with torch.no_grad():
                # batch_size * n_units * n_capacity
                if self.reward_decay_weights is None:
                    self.reward_decay_weights = weights
                else:
                    self.reward_decay_weights = self.reward_decay_weights * self.reward_gamma() + weights
                fresh_ratio = self.reward_decay_weights * self.reward_refresh_rate()
                # n_units * n_capacity
                retain_ratio = 1 - fresh_ratio.sum(0)
                # n_units * n_capacity
                self.rewards *= retain_ratio
                self.rewards += (fresh_ratio * reward.view(-1, 1, 1)).sum(0)

        refresh_memory(weights, key, value)
        refresh_reward(weights, reward)

    def new_episode(self):
        self.reward_decay_weights = None


def main():
    opt = opt_parser.parse_opt()
    n_units = 2
    # opt.dim_attention = 32
    # opt.dim_attention_unit = 8
    # opt.dim_outputs_unit = 9
    # opt.batch_size = 3
    # opt.memory_capacity = 3

    weights = torch.rand(opt.batch_size, n_units, opt.memory_capacity)
    key = torch.rand(opt.batch_size, n_units, opt.dim_attention_unit)
    value = torch.rand(opt.batch_size, n_units, opt.dim_outputs_unit)

    l = UnitWiseMemory(n_units, opt)

    reward = torch.zeros(size=(opt.batch_size,))
    l.refresh(weights, key, value, reward)
    l.refresh(weights, key, value, reward)


if __name__ == '__main__':
    main()
