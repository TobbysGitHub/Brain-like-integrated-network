import torch



def similarity(x1, x2):
    """
    :param x1: batch_size * n_units * d
    :param x2:
    """

    batch_size, n_units, _ = x1.size()

    dot = torch.matmul(x1.unsqueeze(-2), x2.unsqueeze(-1)).view(batch_size, n_units)

