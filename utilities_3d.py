import pyvista as pv
import vtk
import torch
import numpy as np
import os
from tqdm import tqdm

def read_vtk(path, device='cpu', dtype=torch.float32):
    data = pv.read(path)
    points = data.points.astype(np.float32)
    points = torch.tensor(points, device=device).type(dtype)
    
    try:
        weigths = data["radius"].astype(np.float32)
        weigths = torch.tensor(weigths, device=device).type(dtype)
    except KeyError:
        N = points.shape[0]
        weigths = torch.ones((N, 1), device=device).type(dtype) / N

    return points, weigths

def affine_transformation_3d(X, theta_x, theta_y, theta_z, scale, translation, noise, device, dtype):
    translation = torch.tensor(translation, device=device).type(dtype)
    
    Rx = torch.tensor([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x),  np.cos(theta_x)]], device=device).type(dtype)
    
    Ry = torch.tensor([[ np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]], device=device).type(dtype)
    
    Rz = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z),  np.cos(theta_z), 0],
                       [0, 0, 1]], device=device).type(dtype)
    
    R = Rz @ Ry @ Rx
    Y = scale * (R @ X.t()).t() + translation
    Y += noise * torch.randn_like(Y)
    return Y.contiguous()

def plot_samples_3d(x, y, a, b, scale_factor=1.0):
    
    def to_numpy(data):
        if torch.is_tensor(data): return data.detach().cpu().numpy().flatten()
        return data.flatten()

    def to_numpy_3d(data):
        if torch.is_tensor(data): data = data.detach().cpu().numpy()
        if data.shape[1] == 2: data = np.hstack([data, np.zeros((data.shape[0], 1))])
        return data

    x_np = to_numpy_3d(x)
    y_np = to_numpy_3d(y)
    
    ra = np.clip(to_numpy(a), 0.1, 5.0)
    rb = np.clip(to_numpy(b), 0.1, 5.0)

    pl = pv.Plotter(window_size=[1000, 800])
    pl.set_background('white')
    
    pl.enable_eye_dome_lighting()  

    cloud_y = pv.PolyData(y_np)
    cloud_y.point_data['radius'] = rb
    
    geom = pv.Sphere(theta_resolution=6, phi_resolution=6)
    
    pl.add_mesh(
        cloud_y.glyph(scale='radius', geom=geom, factor=scale_factor),
        color='#ff3333',       
        opacity=0.3,         
        label='Target (Rouge)'
    )

    cloud_x = pv.PolyData(x_np)
    cloud_x.point_data['radius'] = ra
    
    pl.add_mesh(
        cloud_x.glyph(scale='radius', geom=geom, factor=scale_factor),
        color='#3366ff',       
        opacity=1.0,           
        label='Source (Bleu)'
    )

    pl.view_isometric()
    pl.show()

def render_flow_gif_3d(history, x_orig, y, threshold=None, filename='flow_3d.gif', points_size=20, opacity_target=0.3, fps=20):
    
    if history is None or len(history) == 0:
        print("No history to render.")
        return
    
    output_dir = "./gifs/"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    print(f"Rendering 3D GIF ({len(history)} frames) to {filepath}...")

    
    def to_numpy_3d(data):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if data.shape[1] == 2:
            data = np.hstack([data, np.zeros((data.shape[0], 1))])
        return data

    x_np_orig = to_numpy_3d(x_orig)
    y_np = to_numpy_3d(y)

    colors = x_np_orig[:, 0]

    all_points = [y_np]
    for state in history:
        all_points.append(to_numpy_3d(state['x']))
    all_points_np = np.vstack(all_points)
    
    bbox_min = all_points_np.min(axis=0)
    bbox_max = all_points_np.max(axis=0)
    center = (bbox_max + bbox_min) / 2
    radius = np.linalg.norm(bbox_max - bbox_min)

    pl = pv.Plotter(off_screen=True, window_size=[800, 800])
    pl.set_background('white') # Fond blanc propre
    
    pl.add_mesh(pv.PolyData(y_np), color=(0.55, 0.55, 0.95), 
                point_size=points_size, render_points_as_spheres=True, 
                opacity=opacity_target,
                label='Target')

    source_mesh = pv.PolyData(x_np_orig)
    source_mesh.point_data['colors'] = colors
    source_mesh.point_data['mass_opacity'] = np.ones(len(x_np_orig))

    actor = pl.add_mesh(source_mesh, 
                        scalars='colors', cmap='hsv',
                        point_size=points_size+2, render_points_as_spheres=True,
                        opacity='mass_opacity',
                        show_scalar_bar=False
                        )

    pl.camera.focal_point = center
    
    pl.camera.position = center + np.array([1.0, -1.0, 1.0]) * radius * 1.5
    pl.camera.up = (0, 0, 1) # L'axe Z pointe vers le haut

    pl.open_gif(filepath, fps=fps)

    iterator = history if len(history) < 200 else tqdm(history, desc="Rendering GIF")

    for i, state in enumerate(iterator):
        current_x = to_numpy_3d(state['x'])
        source_mesh.points = current_x
        
        mass = state['mass']
        ratio = state['ratio']
        
        if threshold is not None:
             keep_mask = (ratio >= threshold)
             current_mass = mass * keep_mask.astype(float)
        else:
             current_mass = mass

        max_mass = current_mass.max() + 1e-6
        normalized_opacity = current_mass / max_mass
        
        source_mesh.point_data['mass_opacity'] = normalized_opacity
        
        if max_mass > 1e-5:
             actor.GetMapper().SetScalarRange(0, 1)

        pl.write_frame()
        
    pl.close()
    print(f"GIF 3D saved : {filepath}")    
    
def save_point_cloud_vtk(x, filename="output.vtk"):

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        
    if x.shape[1] == 2:
        x = np.hstack([x, np.zeros((x.shape[0], 1))])
        
    cloud = pv.PolyData(x)
    
    cloud.save(filename)
    print(f"Saved : {filename}")