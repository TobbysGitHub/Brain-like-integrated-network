import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import tb


def contrastive_loss(enc_outputs, agg_outputs, negatives, weights):
    """
    :param enc_outputs: batch_size * n_units * d, encode_outputs
    :param agg_outputs: batch_size * n_units * d, agg_outputs
    :param negatives: n_neg * n_units * d, memories of encode_outputs
    :param weights: batch_size * n_neg * n_units
    """

    def distance(x1, x2):
        cs = cosine_similarity(x1, x2, dim=-1)
        return torch.acos(cs)

    def sin_distance(x1, x2):
        cs = cosine_similarity(x1, x2, dim=-1)
        return torch.sqrt(1 - torch.pow(torch.relu(cs), 2))

    def cos_distance(x1, x2):
        cs = cosine_similarity(x1, x2, dim=-1)
        theta = torch.acos(cs)
        return 1 - torch.relu(cs) + torch.relu(theta - np.pi / 2)

    enc_enc_dists = distance(enc_outputs.unsqueeze(0), enc_outputs.unsqueeze(1))  # batch_size * batch_size * n_units
    # todo
    enc_agg_cos_dists = cos_distance(enc_outputs, agg_outputs)  # batch_size * n_units
    enc_agg_dists = distance(enc_outputs, agg_outputs)  # batch_size * n_units
    enc_neg_dists = distance(enc_outputs.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    # todo
    agg_neg_sin_dists = sin_distance(agg_outputs.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    enc_neg_sin_dists = sin_distance(enc_outputs.unsqueeze(1), negatives)  # batch_size * n_neg * n_units
    agg_neg_dists = distance(agg_outputs.unsqueeze(1), negatives)  # batch_size * n_neg * n_units

    tb.histogram(enc_enc_dists=enc_enc_dists,
                 enc_neg_dists=enc_neg_dists,
                 enc_agg_cos_dists=enc_agg_cos_dists,
                 enc_agg_dists=enc_agg_dists,
                 agg_neg_dists=agg_neg_dists,
                 agg_neg_sin_dists=agg_neg_sin_dists)

    agg_neg_dist = agg_neg_dists.mean(1)
    agg_neg_sin_dist = agg_neg_sin_dists.mean(1)
    enc_neg_sin_dist = enc_neg_sin_dists.mean(1)

    agg_neg_w_dist = (agg_neg_dists * weights).sum(1)  # batch_size * n_units
    agg_neg_w_sin_dist = (agg_neg_sin_dists * weights).sum(1)  # batch_size * n_units

    # squeeze loss to [-1, 0)
    def squash(z):
        return 1 - torch.exp(-z)

    weight_loss = torch.mean(squash(enc_agg_dists / agg_neg_w_dist))
    # background_loss = 0.5 * (torch.mean(squash(enc_agg_cos_dists / agg_neg_sin_dist)) +
    #                          torch.mean(squash(enc_agg_cos_dists / enc_neg_sin_dist)))
    background_loss = (torch.mean(squash(enc_agg_dists / agg_neg_dist)))
    # background_loss = torch.mean(enc_agg_dists - agg_neg_dist)
    # background_loss = 0.5 * torch.mean(squash(enc_agg_dists / enc_neg_dist) + squash(enc_agg_dists / agg_neg_dist))

    tb.add_scalar(background_loss=background_loss,
                  weight_loss=weight_loss)

    return weight_loss, background_loss


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
