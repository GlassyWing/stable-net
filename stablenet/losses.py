import torch
import numpy as np


def cross_covariance_loss_v2(x, w, k=5):
    """

    :param x: shape. = (n, p)
    :param w: shape. = (n, )
    :return:
    """
    n = x.size(0)

    w = w.view(-1, 1)

    w = w * w  # Ensure that w >= 0
    covariance = x.t() * w.t() @ x / n - ((x.t() @ w) / torch.sum(w)) * (w.t() @ x / n)

    # Remove diagonal elements
    step = 1 + (np.cumprod(x.shape[:-1])).sum()
    covariance.view(-1)[::step] = 0

    return torch.norm(covariance,p='fro')


def cross_covariance_loss(a, w):
    """

    :param a: shape. = (n, feat_dim)
    :param w: shape. = (n, )
    :return:
    """
    n = a.size(0)

    w = w.view(-1, 1)
    w = w * w

    weighted_a = w * a
    mu = torch.mean(weighted_a, dim=0, keepdim=True)  # (1, feat_dim )

    weighted_a_norm = weighted_a - mu  # (n, feat_dim)

    covariance = 1 / (n - 1) * weighted_a_norm.permute(1, 0) @ weighted_a_norm  # (feat_dim, feat_dim)

    # Remove diagonal elements
    step = 1 + (np.cumprod(a.shape[:-1])).sum()
    covariance.view(-1)[::step] = 0

    return torch.norm(covariance, p='fro')

def compute_kernel(x, y):
    """

    :param x: shape. = (n, k)
    :param y: shape. = (m, k)
    :return:
    """
    # (n, m, k) => (n, m)
    numerator = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).mean(2) / float(x.size(1))
    return torch.exp(-numerator)


def compute_mmd(x, y):
    if x.dim() == 4:
        x = x.permute(0, 2, 3, 1).flatten(end_dim=2)
    if y.dim() == 4:
        y = y.permute(0, 2, 3, 1).flatten(end_dim=2)
    x_k = compute_kernel(x, x)
    y_k = compute_kernel(y, y)
    xy_k = compute_kernel(x, y)
    mmd = x_k.mean() + y_k.mean() - 2 * xy_k.mean()
    return mmd