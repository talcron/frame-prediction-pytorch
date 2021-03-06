import argparse
import time as tm

import numpy as np
import torch

# noinspection PyProtectedMember
from matrix_sqrt import *


# Compute error
def compute_error(x, s_x):
    norm_x = torch.sqrt(torch.sum(torch.sum(x * x, dim=1), dim=1))
    error = x - torch.bmm(s_x, s_x)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / norm_x
    return torch.mean(error)


# Create random PSD matrix
def create_symm_matrix(batch_size, dim, num_points, tau, device) -> torch.Tensor:
    x = torch.zeros(batch_size, dim, dim).to(device)
    for i in range(batch_size):
        pts = np.random.randn(num_points, dim).astype(np.float32)
        s_x = np.dot(pts.T, pts) / num_points + tau * np.eye(dim).astype(np.float32)
        x[i, :, :] = torch.from_numpy(s_x)
    print('Creating batch %d, dim %d, pts %d, tau %f, dtype %s' % (batch_size, dim, num_points, tau, device))
    return x


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Matrix square root and its gradient demo')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pts', type=int, default=1000, metavar='N',
                        help='number of points to construct covariance matrix (default: 1000)')
    parser.add_argument('--tau', type=float, default=1.0, metavar='N',
                        help='conditioning by adding to the diagonal (default: 1.0)')
    parser.add_argument('--num-iters', type=int, default=10, metavar='N',
                        help='number of schulz iterations (default: 5)')
    parser.add_argument('--dim', type=int, default=64, metavar='N',
                        help='size of the covariance matrix (default: 64)')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable GPU usage')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu'

    # Create matrix and gradient randomly
    A = create_symm_matrix(batch_size=args.batch_size, dim=args.dim, num_points=args.pts, tau=args.tau, device=device)
    A.requires_grad = True

    dl_dz = torch.randn(args.batch_size, args.dim, args.dim).to(device)
    dl_dz = 0.5 * (dl_dz + dl_dz.transpose(1, 2))

    # Forward + backward with SVD
    # Time: O(n^3), Space: O(n^3)
    print('Singular Value Decomposition (SVD):')
    start = tm.time()
    svd_sA, svd_grad = sqrt_svd_lyap(A, -dl_dz, device=device)
    end = tm.time()
    svd_time = end - start
    svd_error = compute_error(A, svd_sA)
    print('  >> forward + backward time %fs, forward error %s' % (svd_time, svd_error.data.item()))

    # Forward pass with Denman-Beavers iterations (no backward)
    print('Denman-Beavers iterations (%i iters) ' % (args.num_iters))
    start = tm.time()
    sA = sqrt_denman_beavers(A, args.num_iters, device=device)
    end = tm.time()
    error = compute_error(A, sA)
    print('  >> forward time %fs, error %s' % (end - start, error.data.item()))
    print('  >> no backward via autograd')

    # Forward pass with Newton-Schulz (autograd version)
    # Time: O(Tn^2), Space: O(Tn^2), with T iterations
    print('Newton-Schulz iterations (%i iters) ' % (args.num_iters))
    start = tm.time()
    sA = sqrt_newton_schulz_autograd(A, args.num_iters, device=device)
    end = tm.time()
    iter_time = end - start
    error = compute_error(A, sA)
    print('  >> forward: time %fs, error %s' % (end - start, error.data.item()))

    # Backward pass with autograd
    start = tm.time()
    # with torch.autograd.profiler.profile() as prof:
    sA.backward(dl_dz)
    # print(prof)
    end = tm.time()
    iter_time += end - start
    backward_error = svd_grad.dist(A.grad.data)
    print('  >> backward via autograd: time %fs, error %f' % (end - start, backward_error))
    print('  >> speedup over SVD: %.1fx' % (svd_time / iter_time))

    # Forward pass with Newton-Schulz
    # Time: O(Tn^2), Space: O(n^2), with T iterations
    print('Newton-Schulz iterations (foward + backward) (%i iters) ' % (args.num_iters))
    start = tm.time()
    sA = sqrt_newton_schulz(A, args.num_iters)
    end = tm.time()
    iter_time = end - start
    error = compute_error(A, sA)
    print('  >> forward: time %fs, error %s' % (end - start, error))

    # Backward pass with Newton-Schulz
    start = tm.time()
    dl_da = lyap_newton_schulz(sA, dl_dz.data, args.num_iters, device=device)
    end = tm.time()
    iter_time += end - start
    backward_error = svd_grad.dist(dl_da)
    print('  >> backward: time %fs, error %f ' % (end - start, backward_error))
    print('  >> speedup over SVD: %.1fx' % (svd_time / iter_time))
