import torch
from torch.nn.functional import cosine_similarity
from tb import *


def contrastive_loss(sample1, sample2, negatives, weights):
    """
    :param sample1: batch_size * n_units * d
    :param sample2: batch_size * n_units * d
    :param negatives: n_neg * n_units * d
    :param weights: batch_size * n_neg * n_units
    """

    #
    # def dist(x1, x2):
    #     x1_norm, x2_norm = torch.norm(x1, dim=-1), torch.norm(x2, dim=-1)
    #     dot = torch.mul(x1, x2).sum(-1)
    #
    #     cosine = dot / (x1_norm * x2_norm)
    #     return cosine
    #
    # def dist(x1, x2):
    #     return torch.sum(torch.abs(x1 - x2), dim=-1)

    def dist(x1, x2):
        return torch.acos(cosine_similarity(x1, x2, dim=-1))

    pos_dist = dist(sample1, sample2)  # batch_size * n_units
    neg_dist_1 = dist(sample1.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    neg_dist_2 = dist(sample2.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    inner_dist = dist(sample1.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    batch_dist = dist(sample1.unsqueeze(0), sample1.unsqueeze(1))  # batch_size * batch_size * n_units
    s_b = batch_dist.shape[0]
    batch_dist.masked_fill_(mask=(torch.eye(s_b).unsqueeze(-1) == 1), value=0.5)

    background_dist_1 = neg_dist_1.mean(1)
    background_dist_2 = neg_dist_2.mean(1)

    neg_dist_1 *= weights
    neg_dist_1 = neg_dist_1.sum(1)  # batch_size * n_units
    neg_dist_2 *= weights
    neg_dist_2 = neg_dist_2.sum(1)  # batch_size * n_units

    if steps[0] % TENSOR_BOARD_STEPS == 1:
        writer.add_histogram('inner_dist', inner_dist, steps[0])
        writer.add_histogram('batch_dist', batch_dist, steps[0])
        writer.add_histogram('pos_dist', pos_dist, steps[0])
        writer.add_histogram('neg_dist', neg_dist_2, steps[0])

    # squeeze loss to [-1, 0)
    loss = (- torch.exp(-pos_dist / neg_dist_1)
            - torch.exp(-pos_dist / neg_dist_2)).mean()  # batch_size * n_units.mean()
    background_loss = (- torch.exp(-pos_dist / background_dist_1)
                       - torch.exp(-pos_dist / background_dist_2)).mean()  # batch_size * n_units.mean()

    return loss, background_loss


def cosine_loss(x1, x2):
    x1_norm, x2_norm = torch.norm(x1, dim=-1), torch.norm(x2, dim=-1)
    dot = torch.mul(x1, x2).sum(-1)

    cosine = dot / (x1_norm * x2_norm)
    return - cosine.mean()


def main():
    batch_size = 5
    n_units = 4
    dim = 3
    n_neg = 6
    x1 = torch.rand(size=(batch_size, n_units, dim))
    x2 = torch.rand(size=(batch_size, n_units, dim))
    neg = torch.rand(size=(batch_size, n_units, n_neg, dim))
    w = torch.ones(batch_size, n_units, n_neg)


if __name__ == '__main__':
    main()
