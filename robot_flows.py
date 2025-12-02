import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pykeops.torch import LazyTensor
import imageio
import os


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

def wasserstein_flow(x, y, a, b, lr, epsilon, rho, num_snapshots=60, save_history=True):
    
    x = x.clone()
    a = a.flatten()
    b = b.flatten()
    
    Nsteps = max(int(5.0 / lr), 50)
    
    frame_indices = np.unique(np.geomspace(1, Nsteps, num=num_snapshots).astype(int))
    save_frames = set(np.concatenate(([0], frame_indices)))
    
    kappa = rho / (rho + epsilon)
    f = torch.zeros(len(x), device=x.device)
    g = torch.zeros(len(y), device=x.device)
    
    history = []
    
    for i in tqdm(range(Nsteps + 1)):
        
        f, g = sinkhorn_log_keops(x, y, a, b, epsilon, kappa, niter=5, f_init=f, g_init=g)
        v, w = compute_robot_fields(x, y, f, g, a, b, epsilon)
        
        f, g = f.detach(), g.detach()
        
        mass_transported = w.view(-1, 1)
        ratio = mass_transported / a.view(-1, 1)
        
        if i in save_frames and save_history:
            state = {
                'x': x.detach().cpu().numpy(),
                'ratio': ratio.flatten().cpu().numpy(),
                'mass': mass_transported.flatten().cpu().numpy()
            }
            history.append(state)
        
        x = x + lr * ratio * v
    
    if save_history:    
        for i in range(10):
            history.insert(0,history[0])
            history.append(history[-1])
    return x, history

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

def apply_spline_robot(x_final, y, a, b, epsilon, rho, sigma_spline):
    
    kappa = rho / (rho + epsilon)
    
    f, g = sinkhorn_log_keops(x_final, y, a, b, epsilon, kappa, niter=50)
    v, w = compute_robot_fields(x_final, y, f, g, a, b, epsilon)
    
    x_i = LazyTensor(x_final[:, None, :])
    x_j = LazyTensor(x_final[None, :, :])
    
    w_j = LazyTensor(w[None, :, None])
    v_j = LazyTensor(v[None, :, :])
    
    # Noyau Gaussien k(x, y)
    D2_ij = ((x_i - x_j) ** 2).sum(-1)
    K_ij = (-D2_ij / (2 * sigma_spline**2)).exp()
    
    num = (K_ij * w_j * v_j).sum(1)
    
    denom = (K_ij * w_j).sum(1)
    
    v_smooth = num / (denom + 1e-16)
    
    return x_final + v_smooth

def compute_local_structure(x, k_neighbors=20, sigma_scale=1.0):
    
    N, D = x.shape
    
    x_i = LazyTensor(x[:, None, :]) 
    x_j = LazyTensor(x[None, :, :])
    D2_ij = ((x_i - x_j) ** 2).sum(-1) 
    
    knn_indices = D2_ij.argKmin(K=k_neighbors, dim=1)
    
    neighbors = x[knn_indices]
    
    local_mean = neighbors.mean(dim=1, keepdim=True)
    centered = neighbors - local_mean
    
    cov = torch.matmul(centered.transpose(1, 2), centered) / (k_neighbors - 1)
    
    e, v = torch.linalg.eigh(cov)
    
    e = torch.clamp(e, min=1e-6) 
    
    inv_e = 1.0 / (e * sigma_scale**2)
    
    precision_matrix = torch.matmul(v * inv_e.unsqueeze(1), v.transpose(1, 2))
    
    return precision_matrix

def apply_anisotropic_spline_robot(x_final, y, a, b, epsilon, rho, sigma_spline):
    
    kappa = rho / (rho + epsilon)
    f, g = sinkhorn_log_keops(x_final, y, a, b, epsilon, kappa, niter=50)
    v_robot, w = compute_robot_fields(x_final, y, f, g, a, b, epsilon)
    
    precision_matrices = compute_local_structure(x_final, k_neighbors=20, sigma_scale=sigma_spline)
    
    N, D = x_final.shape
    P_flat = precision_matrices.view(N, -1)
    
    x_i = LazyTensor(x_final[:, None, :])      
    x_j = LazyTensor(x_final[None, :, :])      
    
    P_j = LazyTensor(P_flat[None, :, :])       
    
    w_j = LazyTensor(w[None, :, None])        
    v_j = LazyTensor(v_robot[None, :, :])
    
    diff = x_i - x_j 
    
    mahalanobis_dist = (diff | P_j.matvecmult(diff))
    
    K_ij = (-0.5 * mahalanobis_dist).exp()
    
    num = (K_ij * w_j * v_j).sum(1)
    denom = (K_ij * w_j).sum(1)
    
    v_smooth = num / (denom + 1e-16)
    
    return x_final + v_smooth


def smooth_robot_registration(x, y, a, b, mode='affine', epsilon=0.05, rho=10.0, Nsteps=20, num_snapshots=60, save_history=True):
    
    x_orig = x.contiguous()
    y = y.contiguous()
    a = a.flatten().contiguous()
    b = b.flatten().contiguous()
    
    A = torch.eye(2, dtype=torch.float32, device=x_orig.device)
    h = torch.zeros(2, dtype=torch.float32, device=x_orig.device)
    
    f = torch.zeros(len(x_orig), device=x_orig.device, dtype=torch.float32)
    g = torch.zeros(len(y), device=x_orig.device, dtype=torch.float32)
    
    kappa = rho / (rho + epsilon)
    
    frame_indices = np.unique(np.geomspace(1, Nsteps, num=num_snapshots).astype(int))
    save_frames = set(np.concatenate(([0], frame_indices)))
    history = []
    print(save_frames)
    
    for i in tqdm(range(Nsteps + 1)):
        
        z = x_orig @ A.t() + h
        
        f, g = sinkhorn_log_keops(z, y, a, b, epsilon, kappa, niter=10, f_init=f, g_init=g)
        v, w = compute_robot_fields(z, y, f, g, a, b, epsilon)
        
        w_col = w.view(-1, 1)
        target_positions = z + v
        displacement_total = target_positions - x_orig
        
        if mode == 'rigid':
            A, h = solve_rigid_closed_form(x_orig, displacement_total, w_col)
        elif mode == 'affine':
            A, h = solve_affine_closed_form(x_orig, displacement_total, w_col)
            
        if i in save_frames and save_history:
            mass_transported = w.view(-1, 1)
            ratio = mass_transported / a.view(-1, 1)
            
            state = {
                'x': z.detach().cpu().numpy(),
                'ratio': ratio.flatten().cpu().numpy(),
                'mass': mass_transported.flatten().cpu().numpy()
            }
            history.append(state)

    x_final = x_orig @ A.t() + h
    
    for i in range(5):
        history.insert(0,history[0])
    
    return A, h, x_final, history