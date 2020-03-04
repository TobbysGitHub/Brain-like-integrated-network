import numpy as np
import torch
import torch.nn as nn


class RandomMask(nn.Module):
    '''
    以 self.p 的概率将传入的张量的元素填充为 -inf
    '''
    def __init__(self, p=0.1, inplace=True):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("mask probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, x, value=-np.inf):
        mask = torch.rand_like(x) < self.p

        if self.inplace:
            return x.masked_fill_(mask, value)
        else:
            return x.masked_fill(mask, value)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)


def main():
    rm = RandomMask(p=0.5)

    a = torch.ones(4, 4)
    rm(a)

    print(a)


if __name__ == '__main__':
    main()
