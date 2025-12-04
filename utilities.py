from random import choices
from imageio import imread
from matplotlib import pyplot as plt
import torch
import numpy as np
import imageio

def load_image(fname):
    img = imread(fname, mode="F")  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(
        np.linspace(0, 1, A.shape[0]),
        np.linspace(0, 1, A.shape[1]),
        indexing="xy",
    )

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[0, :], x_[1, :], 25 * 500 / len(x_), color, edgecolors="none")
    
def plot_samples(x,y, colors):
    plt.scatter(y[:, 0].cpu(), y[:, 1].cpu(), 25, [(0.55, 0.55, 0.95)])
    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), 25, colors, cmap='hsv')
    #plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    
def affine_transformation(X, theta, scale, translation, noise, device, dtype):
    translation = torch.tensor(translation, device=device).type(dtype)
    R = torch.tensor([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], device=device).type(dtype)
    Y = scale * (R @ X.t()).t() + translation
    Y += noise * torch.randn_like(Y)
    return Y.contiguous()

def render_flow_gif(history, x_orig, y, threshold=None, filename='flow.gif', fps=20):
    
    if history is None or len(history) == 0:
        print("No history to render.")
        return
    
    print(f"Render ({len(history)} frames)...")
    
    x_np_orig = x_orig.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    colors = np.cos(10 * x_np_orig[:, 0]) + np.cos(10 * x_np_orig[:, 1])
    
    xmin = min(x_np_orig[:, 0].min(), y_np[:, 0].min()) - 0.1
    xmax = max(x_np_orig[:, 0].max(), y_np[:, 0].max()) + 0.1
    ymin = min(x_np_orig[:, 1].min(), y_np[:, 1].min()) - 0.1
    ymax = max(x_np_orig[:, 1].max(), y_np[:, 1].max()) + 0.1
    
    frames = []
    
    for state in history:
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Target
        ax.scatter(y_np[:, 0], y_np[:, 1], c= [(0.55, 0.55, 0.95)], s=20)
        
        # Source
        sizes = state['mass']
        if threshold:
            sizes = np.where(state['ratio'] < threshold, 0, sizes)
        
        sizes = 30 * (sizes / (sizes.max() + 1e-6))
        
        ax.scatter(state['x'][:, 0], state['x'][:, 1], c=colors, cmap="hsv", s=sizes, alpha=0.8)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image[:, :, :3])
        
        plt.close(fig)

    imageio.mimsave(f"./gifs/{filename}", frames, fps=fps, loop=0)
    print(f"GIF saved : {filename}")
    
    
def plot_flow(history, x_orig, y, threshold=None, filename='trajectory.png', num_frames=6):
    
    if history is None or len(history) == 0:
        print("Error: No history to plot.")
        return

    x_np_orig = x_orig.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    colors = np.cos(10 * x_np_orig[:, 0]) + np.cos(10 * x_np_orig[:, 1])
    
    xmin = min(x_np_orig[:, 0].min(), y_np[:, 0].min()) - 0.1
    xmax = max(x_np_orig[:, 0].max(), y_np[:, 0].max()) + 0.1
    ymin = min(x_np_orig[:, 1].min(), y_np[:, 1].min()) - 0.1
    ymax = max(x_np_orig[:, 1].max(), y_np[:, 1].max()) + 0.1
    
    total_steps = len(history) - 1
    
    indices = [0, 11, 23, 35, 43, 59]
    
    num_plots = len(indices)
    fig, axes = plt.subplots(2, 3, figsize=(2*num_plots, 6), dpi=100)
    axes = axes.flatten()
    if num_plots == 1: axes = [axes]
    
    for ax, idx in zip(axes, indices):
        state = history[idx]
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        ax.scatter(y_np[:, 0], y_np[:, 1], c=[(0.55, 0.55, 0.95)], s=20, alpha=0.3, label='Target')
        
        sizes = state['mass']
        if threshold:
            sizes = np.where(state['ratio'] < threshold, 0, sizes)
        
        sizes = 30 #* (sizes / (sizes.max() + 1e-6))
        
        ax.scatter(state['x'][:, 0], state['x'][:, 1], c=colors, cmap="hsv", s=sizes, alpha=0.9, edgecolors='k', linewidth=0.1)
        
        percent = int(idx / total_steps * 100)
        ax.set_title(f"{percent}%", fontsize=12)

    plt.tight_layout()
    plt.show()