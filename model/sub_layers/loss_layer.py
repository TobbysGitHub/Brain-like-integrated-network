import torch
import torch.nn as nn


class ContrastiveLossLayer(nn.Module):

    def __init__(self, n_units):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(n_units))

    def forward(self, x1, x2, negs, w):
        """
        :param x1: batch_size * n_units * d
        :param x2:
        :param negs:
        :param w: batch_size * n_units * n_neg
        """
        if x1 is None or x2 is None or negs is None or w is None:
            return None
        batch_size, n_units, dim = x1.size()
        n_neg = negs.size()[1]
        negs = negs.unsqueeze(0).expand(batch_size, n_units, n_neg, dim)

        x1_norm, x2_norm, neg_norm = (torch.norm(x, dim=-1) for x in (x1, x2, negs))

        s_12 = (torch.matmul(x1.unsqueeze(-2), x2.unsqueeze(-1)).view(batch_size, n_units)) / (
                x1_norm * x2_norm)
        # batch_size * n_units
        s_12 /= self.temperature.unsqueeze(0)

        e_12 = torch.exp(s_12)

        s_2n = (torch.matmul(
            x2.view(batch_size, n_units, 1, 1, dim),
            negs.view(batch_size, n_units, n_neg, dim, 1)
        ).view(batch_size, n_units, n_neg)) / (
                       x2_norm.unsqueeze(-1) * neg_norm)
        s_2n /= self.temperature.view(1, n_units, 1)
        e_2n = torch.exp(s_2n)
        sum_e_2n = torch.sum(e_2n * w, dim=-1)

        loss = - torch.log(e_12 / sum_e_2n).sum() / batch_size

        return loss


def main():
    batch_size = 5
    n_units = 4
    dim = 3
    n_neg = 6
    x1 = torch.rand(size=(batch_size, n_units, dim))
    x2 = torch.rand(size=(batch_size, n_units, dim))
    neg = torch.rand(size=(batch_size, n_units, n_neg, dim))
    w = torch.ones(batch_size, n_units, n_neg)
    ContrastiveLossLayer(n_units)(x1, x2, neg, w)


if __name__ == '__main__':
    main()
