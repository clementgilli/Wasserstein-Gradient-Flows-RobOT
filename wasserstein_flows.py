import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def distmat_torch(x, y):
    
    x_norm = torch.sum(x**2, dim=1, keepdim=True) 
    y_norm = torch.sum(y**2, dim=1, keepdim=True)
    cross = x @ y.t() 
    return x_norm + y_norm.t() - 2 * cross

def sinkhorn_log_torch(C, a, b, epsilon, kappa, niter=500, f_init=None, g_init=None, return_P=True):
   
    log_a = torch.log(a[:, None])
    log_b = torch.log(b[None, :])
    
    if f_init is not None:
        f = f_init
    else:
        f = torch.zeros_like(a)
    if g_init is not None:
        g = g_init
    else:
        g = torch.zeros_like(b)
    
    for i in range(niter):
        
        tmp = (g[None, :] - C) / epsilon + log_b
        f = -epsilon * kappa * torch.logsumexp(tmp, dim=1)
        
        tmp = (f[:, None] - C) / epsilon + log_a
        g = -epsilon * kappa * torch.logsumexp(tmp, dim=0)
    
    if return_P:
        P_log = (f[:, None] + g[None, :] - C) / epsilon + log_a + log_b
        P = torch.exp(P_log)
        return P
    else:
        return f, g

def scalar_product(A, B):
    return torch.sum(A * B)

def kl_entropy(P, a, b):
    return torch.sum(P * (torch.log(P + 1e-16) - torch.log(a[:, None] + 1e-16) - torch.log(b[None, :] + 1e-16)))

def kl_mass_a(P, a):
    mass_P = torch.sum(P, dim=1)
    return torch.sum((mass_P - a) * torch.log((mass_P + 1e-16) / (a + 1e-16)))

def kl_mass_b(P, b):
    mass_P = torch.sum(P, dim=0)
    return torch.sum((mass_P - b) * torch.log((mass_P + 1e-16) / (b + 1e-16)))

def sinkhorn_loss(P, x, y, a, b, epsilon, rho):
    C = distmat_torch(x, y) 
    loss = scalar_product(P, C) - epsilon * kl_entropy(P, a, b) - rho * kl_mass_a(P, a) + kl_mass_b(P, b)
    return loss


def wasserstein_flow(x, y, a, b, lr, epsilon, rho, Nsteps=200, threshold=1e-6, exact_loss = True):
    
    x = x.T
    y = y.T
    
    colors = (np.cos(10 * x[:, 0].numpy()) + np.cos(10 * x[:, 1].numpy()))
    xxmin, xxmax, xymin, xymax = x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max()
    ymin, ymax = min(xymin, y[:, 1].min()), max(xymax, y[:, 1].max())
    xmin, xmax = min(xxmin, y[:, 0].min()), max(xxmax, y[:, 0].max())
    
    if not exact_loss:
        x = x.clone().requires_grad_(True)
    
    kappa = rho / (rho + epsilon)
    
    display_times = [int(t / lr) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]
    
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    t_0 = time.time()

    print(f"Start Flow: {Nsteps} steps, lr={lr}, eps={epsilon}, rho={rho}")

    for i in tqdm(range(Nsteps + 1)):
        
        C = distmat_torch(x, y)
        P = sinkhorn_log_torch(C, a, b, epsilon, kappa, niter=50).detach()
        
        if exact_loss:
            mass_transported = torch.sum(P, dim=1, keepdim=True) 
            target_sum = P @ y 
            barycenters = target_sum / (mass_transported + 1e-16)
            
            ratio = mass_transported / a[:, None]
            displacement = x - barycenters
            
            x_current = x.clone()
            mass_current = mass_transported.clone()
            
            grad = ratio * displacement

        else:
            mass_current = torch.sum(P, dim=1, keepdim=True)
            x_current = x.clone()
            
            loss = sinkhorn_loss(P, x, y, a, b, epsilon, rho)
            [grad] = torch.autograd.grad(loss, x)
            grad *= x.shape[0]
        
        x = x - lr * grad

        if i in display_times:
            ax = plt.subplot(2, 3, plot_idx)
            plot_idx += 1
            
            xi = x_current.detach().numpy()
            yi = y.detach().numpy()
            mi = mass_current.flatten().numpy() # Pour la taille des points
            mi = np.where(mi < threshold, 0, mi)
            
            ax.scatter(yi[:, 0], yi[:, 1], c=(0.55, 0.55, 0.95), s=10, alpha=0.3, label='Target')
            
            sizes = 30 * (mi / mi.max())
            
            ax.scatter(xi[:, 0], xi[:, 1], c=colors, cmap="hsv", s=sizes, alpha=0.9)
            
            ax.set_title(f"t = {i*lr:.2f}", fontsize=12)
            ax.set_xlim(xmin - 0.1, xmax + 0.1)
            ax.set_ylim(ymin - 0.1, ymax + 0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    print(f"Done in {time.time() - t_0:.2f}s")
    return x