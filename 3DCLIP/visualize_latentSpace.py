import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from matplotlib.colors import ListedColormap
import pyvista as pv
import torchio as tio
import SimpleITK as sitk
import os
import torch
from tqdm import tqdm
from model import CLIP
from clip_dataloader import clip3d_subjects_withName
import multiprocessing

def visualize_latent_space(embeddings, context_vars, names, proj_type="TSNE", proj_comps=[0,1], context_var="normalized_shape_index", mask_loader=None, mask_key="mask", mask_transform=None):
    """
    Args:
        embeddings: (N, D) latent vectors, numpy array
        context_vars: dict of context variable lists/arrays
        names: list of strings, paths to 3D mask files corresponding to latent vectors
        proj_comps: list of two integers (e.g., [0,2] for "pca 1" vs "pca 3")
        context_var: string, name of the context variable for coloring
        mask_loader: optional custom loader for 3D masks. If None, tries SITK load.
        mask_key: string key in dataset for mask location, if using torchio dataset
        mask_transform: optional callable to apply to mask after loading
    """
    # --- Dimensionality reduction ---
    # proj type must be either "TSNE" or "PCA"
    if proj_type.lower() not in ["tsne", "t-sne", "umap", "u-map", "pca"]:
        raise ValueError("proj_type must be one of 'TSNE', 'UMAP', or 'PCA'")
    if proj_type.lower() in ["tsne", "t-sne"]:
        tsne = TSNE(n_components=n_tsne_components, random_state=42)
        z_proj = tsne.fit_transform(embeddings)
    elif proj_type.lower() in ["umap", "u-map"]:
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        z_proj = umap_reducer.fit_transform(embeddings)
    elif proj_type.lower() in ["pca"]:
        n_components = max(proj_comps)+1
        pca = PCA(n_components=n_components)
        z_proj = pca.fit_transform(z_mu_array)

    x, y = z_proj[:, proj_comps[0]], z_proj[:, proj_comps[1]]

    # --- Coloring setup ---
    var_values = np.array(context_vars[context_var])
    unique_values = np.unique(var_values)
    continuous = len(unique_values) > 10
    cmap = plt.cm.viridis if continuous else ListedColormap(plt.cm.tab10.colors[:len(unique_values)])
    if continuous:
        scatter_c = var_values
    else:
        value_to_int = {v: i for i, v in enumerate(unique_values)}
        scatter_c = np.array([value_to_int[v] for v in var_values])
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9,7))
    sc = ax.scatter(x, y, c=scatter_c, cmap=cmap, s=15, alpha=0.8)
    plt.xlabel(f"{proj_type} {proj_comps[0]+1}")
    plt.ylabel(f"{proj_type} {proj_comps[1]+1}")
    plt.title(f"PCA projection: Components {proj_comps[0]+1} vs {proj_comps[1]+1} (Colored by '{context_var}')")

    # Add colorbar/legend
    if continuous:
        cbar = plt.colorbar(sc)
        cbar.set_label(context_var)
    else:
        handles = [plt.Line2D([0],[0], marker='o', color='w', label=str(v), 
                  markerfacecolor=cmap.colors[i], markersize=8) for i, v in enumerate(unique_values)]
        plt.legend(handles=handles, title=context_var, bbox_to_anchor=(1.05,1), loc='upper left')

    # --- 3D Mask Visualization on Click ---
    #popup_open = [None]
    popup_windows = []

    def load_mask(name):
        """Load mask from path. Can override via mask_loader."""
        if mask_loader is not None:
            return mask_loader(name)
        # Otherwise, auto-detect
        if name.endswith('.nii') or name.endswith('.nii.gz'):
            return sitk.GetArrayFromImage(sitk.ReadImage(name))
        else:
            # Fallback: torchio Subject? Or numpy
            return np.load(name)
    
    def show_3d_mask(mask, mask_path):
        import pyvista as pv
        import numpy as np
        import os

        pv.set_jupyter_backend(None)
        grid = pv.wrap(mask.astype(float))
        try:
            surf = grid.contour([0.5])
        except Exception as e:
            print(f"Surface extraction failed: {e}")
            return

        p = pv.Plotter()
        p.add_mesh(surf,
                color="yellow", 
                style="surface", 
                smooth_shading=True, 
                specular=0.7, 
                specular_power=20, 
                show_edges=True, 
                edge_color="black", 
                line_width=1, 
                lighting=True)
        p.add_axes()
        p.add_text(os.path.basename(mask_path), position='upper_edge', font_size=14, color='black')
        p.set_background('white')
        p.show(auto_close=False)
        p.camera_position = 'iso'

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        mouse_x, mouse_y = event.xdata, event.ydata
        dists = np.hypot(x - mouse_x, y - mouse_y)
        idx = np.argmin(dists)
        threshold = 0.12 * (x.max()-x.min())
        if dists[idx] > threshold:
            return

        mask_path = names[idx]
        mask = load_mask(mask_path)
        if mask_transform:
            mask = mask_transform(mask)
        p = multiprocessing.Process(target=show_3d_mask, args=(mask, mask_path))
        p.start()

    def zoom(event):
        base_scale = 1.2
        ax = event.inaxes
        if ax is None:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        ax.figure.canvas.draw_idle()


    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
#    # Connect scroll event for zooming
    fig.canvas.mpl_connect('scroll_event', zoom)
    plt.tight_layout()
    plt.show()

# --------------------
# Example usage:
# visualize_latent_space(z_mu_array, context_vars, [batch['name'] for batch in ds_val], proj_comps=[0,2], context_var="normalized_shape_index")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_dir_val = "/data/Data/laa_measures/val/quantile_normalized.json"
    data_dir = "/data/Data/cropped_laa128_64mm/masks"
    output_dir = "/storage/experiments/latent_plots/clip3d/quantile_128/"
    n_tsne_components = 3
    n_pca_components = 8
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define data loader 
    ds_val = clip3d_subjects_withName(data_dir, json_dir_val)
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #define model 
    embed_dim = 128                # Standard CLIP embedding dimension
    image_resolution = 128         # Your volume size (128³)
    vision_layers = (3, 4, 6, 3)   # Small ResNet50-like architecture (4 layers)
    vision_width = 64              # Standard width

    context_length = 18            # Number of context tokens (your shape descriptors)
    transformer_width = 128        # Width of transformer model for context encoder
    transformer_heads = 4          # Number of heads (256 dimension with 4 heads, 64 dim per head)
    transformer_layers = 4         # Moderate depth

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width,
        context_length, transformer_width, transformer_heads, transformer_layers
    ).to(device)
    # Load model weights
    model_path = "/data/Data/bjorn/models/clip3d/quantile_128_all/best_clip3d.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    model.eval()  # Set model to evaluation mode

    # Collect data
    z_mu_list = []
    name_list = []
    context_vars = {
        "tortuosity": [],
        "centerline_length": [],
        "max_geodesic_distance": [],
        "volume": [],
        "angle_ostium_laa": [],
        "cl_cut_25_elongation": [],
        "cl_cut_25_cutarea": [],
        "cl_cut_50_elongation": [],
        "cl_cut_50_cutarea": [],
        "cl_cut_75_elongation": [],
        "cl_cut_75_cutarea": [],
        "radii_95": [],
        "normalized_shape_index": [],
        "elongation": [],
        "flatness": [],
        "surface_area": [],
        "ostium_major_axis_length": [],
        "ostium_minor_axis_length": [],
    }

    for batch in tqdm(ds_val):
            images, context_vectors, name = batch['mask'][tio.DATA], batch['context'], batch['name']
            images = images.to(device, non_blocking=True)
            context_vectors = context_vectors.to(device, non_blocking=True)
            with torch.no_grad():
                images = images.unsqueeze(0)  # Add batch dimension if necessary
                context_vectors = context_vectors.unsqueeze(0)
                #encode_images = model.encode_image(images
                encode_images, unpooled_txt_features = model.extract_features(images, context_vectors)
                encode_images = encode_images / encode_images.norm(dim=1, keepdim=True)
                unpooled_txt_features = unpooled_txt_features / unpooled_txt_features.norm(dim=1, keepdim=True)
            
            encode_images = encode_images.squeeze().cpu().numpy()  # Convert to numpy array
            unpooled_txt_features = unpooled_txt_features.squeeze().flatten().cpu().numpy()  # Convert to numpy array
            
            z_mu_list.append(unpooled_txt_features)
            name_withDir = os.path.join(data_dir, name)
            name_list.append(name_withDir)
            context = context_vectors.squeeze().cpu().numpy()
            for i, key in enumerate(context_vars.keys()):
                context_vars[key].append(context[i])



    # Stack z_mu vectors
    z_mu_array = np.vstack(z_mu_list)  # Shape: (n_samples, z_dim)
    visualize_latent_space(z_mu_array, context_vars, name_list, proj_type="pca", proj_comps=[0,1], context_var="normalized_shape_index", mask_loader=None, mask_key="mask", mask_transform=None)