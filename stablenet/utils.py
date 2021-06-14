import torch
import numpy as np


def fourier_mapping(x, k=5):
    """

    :param x: shape. = (bsz, feat_dim)
    :return: tensor: shape. =  (bsz, k x feat_dim)
    """
    bsz, feat_dim = x.shape
    w = torch.randn(size=(feat_dim, feat_dim * k), dtype=x.dtype, device=x.device)
    fi = 2 * torch.rand(size=(bsz, feat_dim * k), dtype=x.dtype, device=x.device) * np.pi

    return 1.414 * torch.cos(x @ w + fi)
