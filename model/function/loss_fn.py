import torch


def contrastive_loss(sample1, sample2, negatives, weights):
    """
    :param sample1: batch_size * n_units * d
    :param sample2: batch_size * n_units * d
    :param negatives: n_units * n_neg * d
    :param weights: batch_size * n_units * n_neg
    """

    def dist(x1, x2):
        return torch.sum(torch.abs(x1 - x2), dim=-1)

    pos_dist = dist(sample1, sample2)  # batch_size * n_units
    neg_dist = dist(sample2.unsqueeze(-2), negatives)  # batch_size * n_units * n_neg

    neg_dist *= weights
    neg_dist = neg_dist.sum(-1)

    # squeeze loss to [-1, 0)
    result = - torch.exp(-pos_dist / neg_dist)  # batch_size * n_units

    return result


def cosine_similarity(x1, x2):
    x1_norm, x2_norm = torch.norm(x1, dim=-1), torch.norm(x2, dim=-1)
    dot = torch.mul(x1, x2).sum(-1)

    similarity = dot / (x1_norm * x2_norm)
    return similarity


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
