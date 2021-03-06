"""
Matrix square root and its gradient on the GPU
Author: Subhransu Maji (smaji@cs.umass.edu)
Date: Dec 19, 2017

Port to python 3.8, pytorch 1.7
Author: Ian Pegg
Date: Mar 5, 2021
"""
import torch

__all__ = ['sqrt_svd_lyap', 'sqrt_denman_beavers', 'sqrt_newton_schulz', 'sqrt_newton_schulz_autograd',
           'lyap_newton_schulz']


# Forward + Backward via SVD decomposition
def sqrt_svd_lyap(x, dl_dz, device):
    batch_size = x.data.shape[0]
    dim = x.data.shape[1]
    dl_da = torch.zeros(batch_size, dim, dim).to(device)
    s_x = torch.zeros(batch_size, dim, dim).to(device)
    for i in range(batch_size):
        U, S, V = x[i, :, :].data.svd()
        s_x[i, :, :] = (U.mm(S.diag().sqrt())).mm(V.t())
        S = S.diag().sqrt().mm(torch.ones(dim, dim).to(device))
        IU = U.t()
        X = -U.mm(
            ((IU.mm(dl_dz[i, :, :].data)).mm(IU.t()))
            / (S + S.t())
        ).mm(U.t())
        dl_da[i, :, :] = X
    return s_x, dl_da


# Forward via Denman-Beavers iterations
def sqrt_denman_beavers(x, num_iterations, device):
    batch_size = x.data.shape[0]
    dim = x.data.shape[1]
    s_x = torch.zeros(batch_size, dim, dim).to(device)
    for n in range(batch_size):
        Y = (x[n, :, :]).data
        Z = torch.eye(dim, dim).to(device)
        for i in range(num_iterations):
            Y_ = 0.5 * (Y + Z.inverse())
            Z = 0.5 * (Z + Y.inverse())
            Y = Y_
        s_x[n, :, :] = Y
    return s_x


# Forward via Newton-Schulz iterations
# Backward via autograd
def sqrt_newton_schulz_autograd(x, num_iterations, device):
    batch_size = x.data.shape[0]
    dim = x.data.shape[1]
    norm_x = x.mul(x).sum(dim=1).sum(dim=1).sqrt()
    Y = x.div(norm_x.view(batch_size, 1, 1).expand_as(x))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).to(device)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).to(device)

    for i in range(num_iterations):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    s_x = Y * torch.sqrt(norm_x).view(batch_size, 1, 1).expand_as(x)
    return s_x


# Forward via Newton-Schulz iterations (non autograd version)
# Seems to be slightly faster and has much lower memory overhead
def sqrt_newton_schulz(x: torch.Tensor, num_iterations=10):
    device = x.device
    if x.ndim == 2:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    dim = x.shape[1]
    norm_x = x.mul(x).sum(dim=1).sum(dim=1).sqrt()
    Y = x.div(norm_x.view(batch_size, 1, 1).expand_as(x))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).to(device)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).to(device)
    for i in range(num_iterations):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    s_x = Y * torch.sqrt(norm_x).view(batch_size, 1, 1).expand_as(x)
    return s_x.squeeze()


# Backward via iterative Lyapunov solver
def lyap_newton_schulz(x, dl_dz, num_iterations, device):
    batch_size = x.shape[0]
    dim = x.shape[1]
    norm_x = x.mul(x).sum(dim=1).sum(dim=1).sqrt()
    Y = x.div(norm_x.view(batch_size, 1, 1).expand_as(x))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).to(device)
    q = dl_dz.div(norm_x.view(batch_size, 1, 1).expand_as(x))
    for i in range(num_iterations):
        q = 0.5 * (q.bmm(3.0 * I - Y.bmm(Y)) - Y.transpose(1, 2).bmm(Y.transpose(1, 2).bmm(q) - q.bmm(Y)))
        Y = 0.5 * Y.bmm(3.0 * I - Y.bmm(Y))
    dl_da = 0.5 * q
    return dl_da
