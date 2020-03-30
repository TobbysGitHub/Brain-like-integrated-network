import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tensor_board import tb


def contrastive_loss(enc_outputs, agg_outputs, att_outputs, negatives, weights):
    """
    :param enc_outputs: batch_size * n_units * d, encode_outputs
    :param agg_outputs: batch_size * n_units * d, agg_outputs
    :param att_outputs: batch_size * n_units * d, att_outputs
    :param negatives: n_neg * n_units * d, memories of encode_outputs
    :param weights: batch_size * n_neg * n_units
    """

    def theta(x1, x2):
        cos = cosine_similarity(x1, x2, dim=-1)
        cos.data = torch.clamp(cos, min=1e-5 - 1, max=1 - 1e-5)
        return torch.acos(cos)

    enc_enc_thetas = theta(enc_outputs.unsqueeze(0),
                           enc_outputs.unsqueeze(1))  # batch_size * batch_size * n_units
    enc_agg_thetas = theta(enc_outputs, agg_outputs)  # batch_size * n_units
    enc_att_thetas = theta(enc_outputs, att_outputs)  # batch_size * n_units
    agg_negs_thetas = theta(agg_outputs.unsqueeze(1), negatives)  # batch_size * n_neg * n_units

    tb.histogram(enc_enc_thetas=enc_enc_thetas,
                 enc_agg_thetas=enc_agg_thetas,
                 enc_att_thetas=enc_att_thetas,
                 agg_negs_thetas=agg_negs_thetas,
                 weights_max=None if weights is None else weights.max(1)[0])

    agg_neg_thetas = agg_negs_thetas.mean(1)
    agg_neg_w_thetas = (agg_negs_thetas * weights).sum(1)

    pre_train_loss = torch.mean(enc_agg_thetas - enc_enc_thetas)
    weight_loss = torch.mean(enc_agg_thetas - agg_neg_w_thetas)
    background_loss = torch.mean(enc_agg_thetas - agg_neg_thetas)

    tb.add_scalar(
        background_loss=background_loss,
        weight_loss=weight_loss)

    return weight_loss, background_loss, pre_train_loss


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
