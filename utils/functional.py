from typing import Tuple

import torch
from utils.matrix_sqrt import sqrt_newton_schulz as sqrtm

__all__ = ['stats', 'sqrtm', 'frechet_inception_distance']


def stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of x.

    Args:
        x: torch tensor of shape (n_samples, cardinality)

    Returns:
        mean, covariance
    """
    n_samples = x.shape[0]
    mean = torch.mean(x, dim=0, keepdim=True)
    c = x - mean
    cv = 1 / (n_samples - 1) * c.T @ c
    return mean.squeeze(), cv


def frechet_inception_distance(fake_embedding: torch.Tensor, real_embedding: torch.Tensor) -> float:
    """
    Given a set of embeddings, compute the frechet inception distance
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance

    Args:
        fake_embedding: Embedding of the generated data
        real_embedding: Embedding of a sample of real data

    Returns:
        FID score
    """
    device = real_embedding.device
    real_mean, real_cov = stats(real_embedding)
    fake_mean, fake_cov = stats(fake_embedding)
    fake_mean, fake_cov = fake_mean.to(device), fake_cov.to(device)

    fid = torch.norm(real_mean - fake_mean, p=2) + \
        torch.trace(real_cov) + torch.trace(fake_cov) - 2 * torch.trace(sqrtm(real_cov @ fake_cov))
    return fid
