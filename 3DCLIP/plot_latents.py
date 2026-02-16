import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm
from clip_dataloader import clip3d_dataloader, clip3d_subjects_dataset  # Assuming you have a dataloader defined in clip_dataloader.py
from model import CLIP  # Assuming you have a CLIP model defined in model.py
import torch
import torchio as tio

# Define directories
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
ds_val = clip3d_subjects_dataset(data_dir, json_dir_val)
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

context_length = 13            # Number of context tokens (your shape descriptors)
transformer_width = 256        # Width of transformer model for context encoder
transformer_heads = 4          # Number of heads (256 dimension with 4 heads, 64 dim per head)
transformer_layers = 4         # Moderate depth

model = CLIP(
    embed_dim,
    image_resolution, vision_layers, vision_width,
    context_length, transformer_width, transformer_heads, transformer_layers
).to(device)
# Load model weights
model_path = "/data/Data/bjorn/models/clip3d/quantile_128/best_clip3d.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model weights loaded from {model_path}")
model.eval()  # Set model to evaluation mode

# Collect data
z_mu_list = []
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
}

for batch in tqdm(ds_val):
    images, context_vectors = batch['mask'][tio.DATA], batch['context']
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        images = images.unsqueeze(0)  # Add batch dimension if necessary
        encode_images = model.encode_image(images)
    encode_images = encode_images.squeeze().cpu().numpy()  # Convert to numpy array
    z_mu_list.append(encode_images)
    context = context_vectors.numpy()
    for i, key in enumerate(context_vars.keys()):
        context_vars[key].append(context[i])


# Stack z_mu vectors
z_mu_array = np.vstack(z_mu_list)  # Shape: (n_samples, z_dim)
print(f"Shape of z_mu_array: {z_mu_array.shape}")

# Reduce dimensionality with PCA before applying T-SNE
pca_components = min(n_pca_components, z_mu_array.shape[1])  # Choose 50 or the number of features, whichever is smaller
pca = PCA(n_components=pca_components)
z_mu_pca = pca.fit_transform(z_mu_array)
print(f"Shape of z_mu_pca: {z_mu_pca.shape}")

# Perform T-SNE
tsne = TSNE(n_components=n_tsne_components, random_state=42)
z_mu_tsne = tsne.fit_transform(z_mu_pca)

# Convert context variables to numpy arrays
for key in context_vars:
    context_vars[key] = np.array(context_vars[key])

# Plotting
for i in range(1, n_tsne_components):
    for var_name, var_values in context_vars.items():
        unique_values = np.unique(var_values)
        if len(unique_values) >= 10:
            # Proceed with quantile binning
            percentiles = np.arange(0, 101, 10)  # 0% to 100% inclusive, step 10%
            quantile_edges = np.percentile(var_values, percentiles)
            # Remove duplicates in quantile_edges
            quantile_edges = np.unique(quantile_edges)
            # If we have fewer than 2 quantile edges, we cannot bin
            if len(quantile_edges) < 2:
                print(f"Variable {var_name} cannot be binned into quantiles due to lack of variability.")
                # Proceed to plot actual values
                plt.figure(figsize=(8,6))
                scatter = plt.scatter(z_mu_tsne[:,i-1], z_mu_tsne[:,i], c=var_values, cmap='viridis', s=5)
                cbar = plt.colorbar(scatter)
                cbar.set_label(var_name)
                plt.title(f'T-SNE projection colored by {var_name}')
                plt.xlabel(f'T-SNE Component {i}')
                plt.ylabel(f'T-SNE Component {i+1}')
                # Save plot
                output_path = os.path.join(output_dir, f'tsne_{var_name}_c{i}v{i+1}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
            else:
                # Assign bins
                quantile_bins = np.digitize(var_values, quantile_edges, right=True) - 1  # Subtract 1 to make bins start from 0
                # Ensure bins are within valid range
                quantile_bins[quantile_bins >= len(quantile_edges) - 1] = len(quantile_edges) - 2
                # Create labels
                quantile_labels = [f'{quantile_edges[i]:.2f}-{quantile_edges[i+1]:.2f}' for i in range(len(quantile_edges)-1)]
                # Now, plot
                plt.figure(figsize=(8,6))
                cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(quantile_labels))))
                scatter = plt.scatter(z_mu_tsne[:,i-1], z_mu_tsne[:,i], c=quantile_bins, cmap=cmap, s=5)
                cbar = plt.colorbar(scatter, ticks=np.arange(len(quantile_labels)))
                cbar.ax.set_yticklabels(quantile_labels)
                cbar.set_label('Quantile Bins')
                plt.title(f'T-SNE projection colored by quantiles of {var_name}')
                plt.xlabel(f'T-SNE Component {i}')
                plt.ylabel(f'T-SNE Component {i+1}')
                # Save plot
                output_path = os.path.join(output_dir, f'tsne_{var_name}_quantiles_c{i}v{i+1}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
        else:
            # Variable has few unique values; use categorical colors
            plt.figure(figsize=(8,6))
            # For categorical variables, we can map the unique values to integers for color-coding
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            int_values = np.array([value_to_int[v] for v in var_values])
            # Define a discrete colormap
            cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_values)])
            scatter = plt.scatter(z_mu_tsne[:,i-1], z_mu_tsne[:,i], c=int_values, cmap=cmap, s=5)
            # Create legend
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(v), markerfacecolor=cmap.colors[i], markersize=5)
                    for i, v in enumerate(unique_values)]
            plt.legend(handles=handles, title=var_name)
            plt.title(f'T-SNE projection colored by {var_name}')
            plt.xlabel(f'T-SNE Component {i}')
            plt.ylabel(f'T-SNE Component {i+1}')
            # Save plot
            output_path = os.path.join(output_dir, f'tsne_{var_name}_c{i}v{i+1}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()

for i in range(1,n_pca_components):
    for var_name, var_values in context_vars.items():
        unique_values = np.unique(var_values)
        if len(unique_values) >= 10:
            # Proceed with quantile binning
            percentiles = np.arange(0, 101, 10)  # 0% to 100% inclusive, step 10%
            quantile_edges = np.percentile(var_values, percentiles)
            # Remove duplicates in quantile_edges
            quantile_edges = np.unique(quantile_edges)
            # If we have fewer than 2 quantile edges, we cannot bin
            if len(quantile_edges) < 2:
                print(f"Variable {var_name} cannot be binned into quantiles due to lack of variability.")
                # Proceed to plot actual values
                plt.figure(figsize=(8,6))
                scatter = plt.scatter(z_mu_pca[:,i-1], z_mu_pca[:,i], c=var_values, cmap='viridis', s=5)
                cbar = plt.colorbar(scatter)
                cbar.set_label(var_name)
                plt.title(f'T-SNE projection colored by {var_name}')
                plt.xlabel(f'T-SNE Component {i}')
                plt.ylabel(f'T-SNE Component {i+1}')
                # Save plot
                output_path = os.path.join(output_dir, f'pca_{var_name}_c{i}v{i+1}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
            else:
                # Assign bins
                quantile_bins = np.digitize(var_values, quantile_edges, right=True) - 1  # Subtract 1 to make bins start from 0
                # Ensure bins are within valid range
                quantile_bins[quantile_bins >= len(quantile_edges) - 1] = len(quantile_edges) - 2
                # Create labels
                quantile_labels = [f'{quantile_edges[i]:.2f}-{quantile_edges[i+1]:.2f}' for i in range(len(quantile_edges)-1)]
                # Now, plot
                plt.figure(figsize=(8,6))
                cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(quantile_labels))))
                scatter = plt.scatter(z_mu_pca[:,i-1], z_mu_pca[:,i], c=quantile_bins, cmap=cmap, s=5)
                cbar = plt.colorbar(scatter, ticks=np.arange(len(quantile_labels)))
                cbar.ax.set_yticklabels(quantile_labels)
                cbar.set_label('Quantile Bins')
                plt.title(f'PCA projection colored by quantiles of {var_name}')
                plt.xlabel(f'PCA Component {i}')
                plt.ylabel(f'PCA Component {i+1}')
                # Save plot
                output_path = os.path.join(output_dir, f'pca_{var_name}_quantiles_c{i}v{i+1}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
        else:
            # Variable has few unique values; use categorical colors
            plt.figure(figsize=(8,6))
            # For categorical variables, we can map the unique values to integers for color-coding
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            int_values = np.array([value_to_int[v] for v in var_values])
            # Define a discrete colormap
            cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_values)])
            scatter = plt.scatter(z_mu_pca[:,i-1], z_mu_pca[:,i], c=int_values, cmap=cmap, s=5)
            # Create legend
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(v), markerfacecolor=cmap.colors[i], markersize=5)
                    for i, v in enumerate(unique_values)]
            plt.legend(handles=handles, title=var_name)
            plt.title(f'PCA projection colored by {var_name}')
            plt.xlabel(f'PCA Component {i}')
            plt.ylabel(f'PCA Component {i+1}')
            # Save plot
            output_path = os.path.join(output_dir, f'pca_{var_name}_c{i}v{i+1}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
