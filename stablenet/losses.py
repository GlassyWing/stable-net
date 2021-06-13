import torch
import numpy as np


def cross_covariance_loss(a, w):
    """

    :param a: shape. = (n, feat_dim)
    :param w: shape. = (n, )
    :return:
    """
    n = a.size(0)

    w = w.view(-1, 1)

    weighted_a = w * a
    mu = torch.mean(weighted_a, dim=0, keepdim=True)  # (1, feat_dim )

    weighted_a_norm = weighted_a - mu  # (n, feat_dim)

    covariance = 1 / (n - 1) * weighted_a_norm.permute(1, 0) @ weighted_a_norm  # (feat_dim, feat_dim)

    # Remove diagonal elements
    step = 1 + (np.cumprod(a.shape[:-1])).sum()
    covariance.view(-1)[::step] = 0

    return torch.norm(covariance, p='fro')
