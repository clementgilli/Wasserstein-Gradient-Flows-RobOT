import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pykeops.torch import LazyTensor


def sinkhorn_log_keops(x, y, a, b, epsilon, kappa, niter=50, f_init=None, g_init=None):
    
    N, D = x.shape
    M, _ = y.shape

    if f_init is not None:
        f = f_init
        g = g_init
    else:
        f = torch.zeros(N, device=x.device, dtype=x.dtype)
        g = torch.zeros(M, device=x.device, dtype=x.dtype)

    log_a = torch.log(a)
    log_b = torch.log(b)
    
    x_i = LazyTensor(x[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(y[None, :, :])  # (1, M, D)
    
    for i in range(niter):
          
        g_j = LazyTensor(g[None, :, None])  # (1, M, 1) pour KeOps
        SqDist = ((x_i - y_j) ** 2).sum(-1)
        to_reduce = (g_j - SqDist) / epsilon + LazyTensor(log_b[None, :, None])
        lse_f = to_reduce.logsumexp(1).flatten()
        
        f = -epsilon * kappa * lse_f
        f_i = LazyTensor(f[:, None, None]) # (N, 1, 1)
        to_reduce = (f_i - SqDist) / epsilon + LazyTensor(log_a[:, None, None])
        lse_g = to_reduce.logsumexp(0).flatten()
        
        g = -epsilon * kappa * lse_g
        
    return f, g

def compute_robot_fields(x, y, f, g, a, b, epsilon):
    
    x, y = x, y
    f, g = f, g
    a, b = a, b
    
    f = f.view(-1, 1)
    g = g.view(-1, 1)
    a = a.view(-1, 1) 
    b = b.view(-1, 1) 
    
    x_i = LazyTensor(x, axis=0)      
    y_j = LazyTensor(y, axis=1)      
    f_i = LazyTensor(f, axis=0)      
    g_j = LazyTensor(g, axis=1)
    
    log_a_i = LazyTensor(a.log(), axis=0)
    log_b_j = LazyTensor(b.log(), axis=1)

    SqDist = ((x_i - y_j) ** 2).sum(-1)
    
    S_ij = (f_i + g_j - SqDist) / epsilon + log_a_i + log_b_j
    
    log_w_i = S_ij.logsumexp(1)
    w_i = log_w_i.exp()
    
    Pi_ij = S_ij.exp()
    w_y = (Pi_ij * y_j).sum(1)
    
    v_i = (w_y / (w_i + 1e-16)) - x
    
    return v_i.float(), w_i.flatten().float()

def wasserstein_flow_fast(x, y, a, b, lr, epsilon, rho, threshold=None):
    
    a = a.flatten()
    b = b.flatten()
    
    colors = (np.cos(10 * x[:, 0].cpu().numpy()) + np.cos(10 * x[:, 1].cpu().numpy()))
    xxmin, xxmax = min(x[:, 0].min(), y[:, 0].min()).item(), max(x[:, 0].max(), y[:, 0].max()).item()
    yymin, yymax = min(x[:, 1].min(), y[:, 1].min()).item(), max(x[:, 1].max(), y[:, 1].max()).item()
    
    display_times = [int(t / lr) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]
    Nsteps = display_times[-1]
    
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    t_0 = time.time()
    
    kappa = rho / (rho + epsilon)
    
    f = torch.zeros(len(x), device=x.device)
    g = torch.zeros(len(y), device=x.device)
    
    for i in tqdm(range(Nsteps+1)):
        
        f, g = sinkhorn_log_keops(x, y, a, b, epsilon, kappa, niter=10, f_init=f, g_init=g)
        
        v, w = compute_robot_fields(x, y, f, g, a,b,epsilon)
        
        f = f.detach()
        g = g.detach()
        
        mass_transported = w.view(-1, 1)
        ratio = mass_transported / a.view(-1, 1)
        
        x_current = x.clone()
        mass_current = mass_transported.clone()
        
        grad = - ratio * v
        
        x = x - lr * grad
        
        if i in display_times:
            ax = plt.subplot(2, 3, plot_idx)
            plot_idx += 1
            
            xi = x_current.detach().cpu().numpy()
            yi = y.detach().cpu().numpy()
            mi = mass_current.flatten().cpu().numpy()
            if threshold:
                mi = np.where(ratio.flatten().cpu().numpy() < threshold, 0, mi)
            
            ax.scatter(yi[:, 0], yi[:, 1], c=[(0.55, 0.55, 0.95)], s=30, label='Target')
            
            sizes = 30 * (mi / mi.max())
            
            ax.scatter(xi[:, 0], xi[:, 1], c=colors, cmap="hsv", s=sizes, alpha=0.9)
            
            ax.set_title(f"t = {i*lr:.2f}", fontsize=12)
            ax.set_xlim(xxmin - 0.1, xxmax + 0.1)
            ax.set_ylim(yymin - 0.1, yymax + 0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    xi = x.detach().cpu().numpy()
    yi = y.detach().cpu().numpy()
    ax.scatter(yi[:, 0], yi[:, 1], c=[(0.55, 0.55, 0.95)], s=30, label='Target')
    ax.scatter(xi[:, 0], xi[:, 1], c=colors, cmap="hsv", s=30, alpha=0.9)
    ax.set_xlim(xxmin - 0.1, xxmax + 0.1)
    ax.set_ylim(yymin - 0.1, yymax + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Final Result", fontsize=14)
    plt.show()
    
    print(f"Total time: {time.time() - t_0:.2f} seconds")
    return x


def solve_rigid_closed_form(x, v, w):
    
    w_sum = w.sum()
    x_c = (x * w).sum(dim=0) / w_sum
    
    y_target = x + v
    y_c = (y_target * w).sum(dim=0) / w_sum
    
    X_centered = x - x_c
    Y_centered = y_target - y_c
    
    H = (X_centered * w).t() @ Y_centered
    
    U, S, Vh = torch.linalg.svd(H)
    
    R = Vh.t() @ U.t()
    
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.t() @ U.t()
    
    t = y_c - (R @ x_c)
    
    # Transformation: x @ R.T + t
    return R, t

def solve_affine_closed_form(x, v, w):
    
    w_sum = w.sum()
    x_c = (x * w).sum(dim=0) / w_sum
    y_target = x + v
    y_c = (y_target * w).sum(dim=0) / w_sum
    
    X_centered = x - x_c
    Y_centered = y_target - y_c
     
    Cov_XX = (X_centered * w).t() @ X_centered
    Cov_XY = (X_centered * w).t() @ Y_centered
    
    # Ridge
    Cov_XX.diagonal().add_(1e-6)
     
    #torch.linalg.solve(A, B)  => AX = B
    A_transpose = torch.linalg.solve(Cov_XX, Cov_XY)
    
    A = A_transpose.t()
    
    t = y_c - (A @ x_c)
    
    return A, t

def apply_spline_robot(x, v, w, sigma_spline):
    
    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])
    
    w_j = LazyTensor(w[None, :, None])
    v_j = LazyTensor(v[None, :, :])
    
    # Noyau Gaussien k(x, y)
    D2_ij = ((x_i - x_j) ** 2).sum(-1)
    K_ij = (-D2_ij / (2 * sigma_spline**2)).exp()
    
    num = (K_ij * w_j * v_j).sum(1)
    
    denom = (K_ij * w_j).sum(1)
    
    v_smooth = num / (denom + 1e-16)
    
    return x + v_smooth


def smooth_robot_registration(x, y, a, b, mode='affine', smooth = True, epsilon=0.05, rho=10.0, Nsteps=20):
    
    x_orig = x.contiguous()
    y = y.contiguous()
    
    a = a.flatten().contiguous()
    b = b.flatten().contiguous()
    
    A = torch.eye(2, dtype=torch.float32, device=x_orig.device)
    h = torch.zeros(2, dtype=torch.float32, device=x_orig.device)
    
    f = torch.zeros(len(x_orig), device=x_orig.device, dtype=torch.float32)
    g = torch.zeros(len(y), device=x_orig.device, dtype=torch.float32)
    
    kappa = rho / (rho + epsilon)
    
    display_times = [0,1,2,3,5, Nsteps]
    
    colors = (np.cos(10 * x_orig[:, 0].cpu().numpy()) + np.cos(10 * x_orig[:, 1].cpu().numpy()))
    
    xxmin, xxmax = min(x_orig[:, 0].min().item(), y[:, 0].min().item()), max(x_orig[:, 0].max().item(), y[:, 0].max().item())
    yymin, yymax = min(x_orig[:, 1].min().item(), y[:, 1].min().item()), max(x_orig[:, 1].max().item(), y[:, 1].max().item())

    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    t0 = time.time()
    
    for i in tqdm(range(Nsteps+1)):
        
        z = x_orig @ A.t() + h
        
        f, g = sinkhorn_log_keops(z, y, a, b, epsilon, kappa, niter=10, f_init=f, g_init=g)
        v, w = compute_robot_fields(z, y, f, g, a, b, epsilon)
        
        w_col = w.view(-1, 1) # (N, 1)
        
        target_positions = z + v
        displacement_total = target_positions - x_orig
        
        if mode == 'rigid':
            A, h = solve_rigid_closed_form(x_orig, displacement_total, w_col)
        elif mode == 'affine':
            A, h = solve_affine_closed_form(x_orig, displacement_total, w_col)
            
        if i in display_times:
            ax = plt.subplot(2, 3, plot_idx)
            plot_idx += 1
            
            zi_disp = z.detach().cpu().numpy()
            yi_disp = y.detach().cpu().numpy()
            
            ax.scatter(yi_disp[:, 0], yi_disp[:, 1], c=[(0.55, 0.55, 0.95)], s=30, alpha=0.3, label='Target')
            
            sizes = 10
            ax.scatter(zi_disp[:, 0], zi_disp[:, 1], c=colors, cmap="hsv", s=30, alpha=0.9)
            
            det_A = torch.det(A).item()
            ax.set_title(f"Iter {i}", fontsize=10)
            ax.set_xlim(xxmin - 0.1, xxmax + 0.1)
            ax.set_ylim(yymin - 0.1, yymax + 0.1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

    x_final = x_orig @ A.t() + h
    plt.tight_layout()
    plt.show()
    
    if smooth:
        z = x_orig @ A.t() + h
        f, g = sinkhorn_log_keops(z, y, a, b, epsilon, kappa, niter=50, f_init=f, g_init=g)
        v, w = compute_robot_fields(z, y, f, g, a, b, epsilon)
        x_final = apply_spline_robot(z, v, w, sigma_spline=0.1)
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    xi = x_final.detach().cpu().numpy()
    yi = y.detach().cpu().numpy()
    ax.scatter(yi[:, 0], yi[:, 1], c=[(0.55, 0.55, 0.95)], s=30, label='Target')
    ax.scatter(xi[:, 0], xi[:, 1], c=colors, cmap="hsv", s=30, alpha=0.9)
    ax.set_xlim(xxmin - 0.1, xxmax + 0.1)
    ax.set_ylim(yymin - 0.1, yymax + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Final Registration Result", fontsize=14)
    plt.show()
    
    print(f"Total time: {time.time() - t0:.2f} seconds")
    
    return A, h, x_final