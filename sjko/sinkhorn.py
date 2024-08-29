import torch
from .utils import *

def softmin(eps, C, f):
    return -eps * (f.view(1, -1) - C / eps).logsumexp(1).view(-1)

def sinkhorn_loop(µ1, µ2, C_xx, C_yy, C_xy, eps_list, init=None, tol=1e-2, iter_counter=torch.zeros(1)):
    C_yx = C_xy.T
    logµ1 = clampedlog(µ1)
    logµ2 = clampedlog(µ2)
    torch.autograd.set_grad_enabled(False)
    eps = eps_list[0]
    if init is None:
        g_12 = softmin(eps, C_yx, logµ1)
        f_21 = softmin(eps, C_xy, logµ2)
        f_11 = softmin(eps, C_xx, logµ1)
        g_22 = softmin(eps, C_yy, logµ2)
    else:
        f_11, g_22, g_12, f_21 = init

    for i, eps in enumerate(eps_list):
        ft_ba = softmin(eps, C_xy, logµ2 + g_12 / eps)
        gt_ab = softmin(eps, C_yx, logµ1 + f_21 / eps)
        ft_aa = softmin(eps, C_xx, logµ1 + f_11 / eps)
        gt_bb = softmin(eps, C_yy, logµ2 + g_22 / eps)
        
        iter_counter[0] += 1
        if max(sqnorm(ft_ba-f_21).item(),
               sqnorm(gt_ab - g_12).item()) < tol**2:
            break

        # Symmetrized updates - see Fig. 3.24.b in Jean Feydy's PhD thesis:
        f_21, g_12 = 0.5 * (f_21 + ft_ba), 0.5 * (g_12 + gt_ab)  # OT(a,b) wrt. a, b
        f_11, g_22 = 0.5 * (f_11 + ft_aa), 0.5 * (g_22 + gt_bb)  # OT(a,a), OT(b,b)
    # As a very last step, we perform a final "Sinkhorn" iteration.
    # As detailed above (around "torch.autograd.set_grad_enabled(False)"),
    # this allows us to retrieve correct expressions for the gradient
    # without having to backprop through the whole Sinkhorn loop.
    torch.autograd.set_grad_enabled(True)

    f_21, g_12 = (
        softmin(eps, C_xy, (logµ2 + g_12 / eps).detach()),
        softmin(eps, C_yx, (logµ1 + f_21 / eps).detach()),
    )
    f_11 = softmin(eps, C_xx, (logµ1 + f_11 / eps).detach())
    g_22 = softmin(eps, C_yy, (logµ2 + g_22 / eps).detach())

    return f_11, g_22, g_12, f_21

def self_transport(µ, c, eps, init = None, tol=1e-4, maxiter=5, iter_counter=torch.zeros(1)):
    logµ = clampedlog(µ)
    if init is None:
        f_µ = softmin(eps, c, logµ)
    else:
        f_µ = init
    k = 0
    d = tol + 1
    while k < maxiter and d > tol:
        k+=1
        f_aa_ = f_µ.clone()
        f_µ = .5*(f_aa_ + softmin(eps, c, logµ + f_µ / eps))
        d = sqnorm(f_µ-f_aa_).item()
    iter_counter[0] += k
    return f_µ