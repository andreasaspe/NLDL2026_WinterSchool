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
import multiprocessing
from mpl_toolkits.mplot3d import proj3d

# NEW: clustering imports
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

def _parse_clustering_type(clustering_type):
    """
    Parse strings like:
      'kmeans:k=6'
      'agglomerative:n_clusters=7,linkage=ward'
      'spectral:n_clusters=8,n_neighbors=12'
      'dbscan:eps=0.4,min_samples=10'
      'hdbscan:min_cluster_size=12'
    Returns algo (lowercased) and dict of params with basic type casting.
    """
    if clustering_type is None or str(clustering_type).strip() == "":
        return "kmeans", {"k": 5}

    text = str(clustering_type).strip()
    parts = text.split(":", 1)
    algo = parts[0].strip().lower()
    params = {}

    if len(parts) == 2 and parts[1].strip():
        for kv in parts[1].split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k, v = k.strip(), v.strip()
                # basic type casting
                if v.isdigit():
                    v_cast = int(v)
                else:
                    try:
                        v_cast = float(v)
                        # if it cleanly casts to float but also to int (e.g., "6.0"), keep float
                    except ValueError:
                        if v.lower() in ["true", "false"]:
                            v_cast = (v.lower() == "true")
                        else:
                            v_cast = v
                params[k] = v_cast
            else:
                params[kv.strip()] = True
    return algo, params

def _cluster_embeddings(embeddings, algo, params):
    """
    Run clustering in the original latent space.
    Returns labels (np.array of ints) and a short description for plotting.
    """
    algo_l = algo.lower()
    labels = None
    desc = ""
    if algo_l in ["kmeans", "k-means"]:
        k = int(params.get("k", params.get("n_clusters", 5)))
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(embeddings)
        desc = f"KMeans (k={k})"
    elif algo_l in ["agglomerative", "agg", "hierarchical"]:
        k = int(params.get("n_clusters", params.get("k", 5)))
        linkage = params.get("linkage", "ward")
        # metric/affinity compatibility across sklearn versions
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=params.get("metric", "euclidean"))
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=params.get("metric", "euclidean"))
        labels = model.fit_predict(embeddings)
        desc = f"Agglomerative (k={k}, linkage={linkage})"
    elif algo_l in ["spectral"]:
        k = int(params.get("n_clusters", params.get("k", 5)))
        if "n_neighbors" in params:
            model = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=int(params["n_neighbors"]),
                assign_labels=str(params.get("assign_labels", "kmeans")),
                random_state=42,
            )
        else:
            model = SpectralClustering(
                n_clusters=k,
                affinity=str(params.get("affinity", "rbf")),
                assign_labels=str(params.get("assign_labels", "kmeans")),
                random_state=42,
            )
        labels = model.fit_predict(embeddings)
        desc = f"Spectral (k={k})"
    elif algo_l in ["dbscan"]:
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embeddings)
        n_core = len(set(labels)) - (1 if -1 in labels else 0)
        desc = f"DBSCAN (eps={eps}, min_samples={min_samples}, clusters={n_core}{', noise' if -1 in labels else ''})"
    elif algo_l in ["hdbscan"]:
        try:
            import hdbscan
        except Exception as e:
            raise ImportError("Requested 'hdbscan' but the package is not installed. Please `pip install hdbscan`.") from e
        min_cluster_size = int(params.get("min_cluster_size", 5))
        min_samples = params.get("min_samples", None)
        if min_samples is not None:
            min_samples = int(min_samples)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = model.fit_predict(embeddings)
        n_core = len(set(labels)) - (1 if -1 in labels else 0)
        desc = f"HDBSCAN (min_cluster_size={min_cluster_size}, clusters={n_core}{', noise' if -1 in labels else ''})"
    else:
        raise ValueError(f"Unknown clustering algorithm: '{algo}'")

    return np.asarray(labels), desc

def visualize_latent_space3D(
    embeddings, context_vars, names,
    proj_type="PCA", proj_comps=[0,1,2],
    clustering_type="kmeans:k=5",
    mask_loader=None, mask_transform=None):

    # --- Dimensionality reduction ---
    proj_type_l = proj_type.lower()
    if proj_type_l not in ["tsne", "t-sne", "umap", "u-map", "pca"]:
        raise ValueError("proj_type must be one of 'TSNE', 'UMAP', or 'PCA'")

    n_proj_comps = len(proj_comps)
    if n_proj_comps != 3:
        raise ValueError("For 3D visualization, proj_comps must be of length 3 (e.g., [0,1,2])")

    if proj_type_l in ["tsne", "t-sne"]:
        reducer = TSNE(n_components=3, random_state=42)
        z_proj = reducer.fit_transform(embeddings)
    elif proj_type_l in ["umap", "u-map"]:
        reducer = umap.UMAP(n_components=3, random_state=42)
        z_proj = reducer.fit_transform(embeddings)
    elif proj_type_l == "pca":
        n_components = max(proj_comps)+1
        pca = PCA(n_components=n_components)
        z_proj = pca.fit_transform(embeddings)

    x, y, z = z_proj[:, proj_comps[0]], z_proj[:, proj_comps[1]], z_proj[:, proj_comps[2]]

    # --- Clustering in original latent space ---
    algo, params = _parse_clustering_type(clustering_type)
    labels, algo_desc = _cluster_embeddings(embeddings, algo, params)

    # Optionally compute silhouette (on non-noise labels)
    sil_text = ""
    try:
        lbl_mask = labels >= 0
        uniq = np.unique(labels[lbl_mask])
        if lbl_mask.any() and len(uniq) >= 2:
            sil = silhouette_score(embeddings[lbl_mask], labels[lbl_mask])
            sil_text = f" | silhouette={sil:.3f}"
    except Exception:
        pass

    # --- Coloring setup (categorical by cluster label) ---
    unique_labels = sorted(np.unique(labels))
    # remap labels to 0..K-1 for colormap indexing
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    scatter_c = np.array([label_to_idx[lab] for lab in labels])

    K = len(unique_labels)
    cmap = ListedColormap(plt.cm.get_cmap('tab20', max(K, 1)).colors[:K])

    # --- Information ---
    print(f"Projection type: {proj_type.upper()}")
    print(f"Components used for projection: {proj_comps}")
    print(f"Clustering: {algo_desc}{sil_text}")
    print("Use left-click to rotate, right-click+drag to zoom, and middle-click to show 3D mask, Q to end application.")

    # --- 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=scatter_c, cmap=cmap, s=20, alpha=0.85)

    ax.set_xlabel(f"{proj_type.upper()} {proj_comps[0]+1}")
    ax.set_ylabel(f"{proj_type.upper()} {proj_comps[1]+1}")
    ax.set_zlabel(f"{proj_type.upper()} {proj_comps[2]+1}")
    plt.title(f"{proj_type.upper()} projection: Components {proj_comps[0]+1} vs {proj_comps[1]+1} vs {proj_comps[2]+1}\n"
              f"(Colored by {algo_desc} labels)")

    # Legend (up to 20 entries to avoid clutter)
    if K <= 20:
        handles = []
        for lab in unique_labels:
            idx = label_to_idx[lab]
            color = cmap.colors[idx]
            lab_str = "Noise (-1)" if lab == -1 else f"Cluster {lab}"
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=lab_str,
                           markerfacecolor=color, markersize=8)
            )
        ax.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.1, 1), loc='upper left')

    def load_mask(name):
        if mask_loader is not None:
            return mask_loader(name)
        if name.endswith('.nii') or name.endswith('.nii.gz'):
            return sitk.GetArrayFromImage(sitk.ReadImage(name))
        else:
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
                   color="yellow", style="surface", smooth_shading=True,
                   specular=0.7, specular_power=20, show_edges=True,
                   edge_color="black", line_width=1, lighting=True)
        p.add_axes()
        p.add_text(os.path.basename(mask_path), position='upper_edge', font_size=14, color='black')
        p.set_background('white')
        p.show(auto_close=False)
        p.camera_position = 'iso'

    def on_scrool_wheel_click(event):
        if event.button != 2:
            return
        if event.inaxes != ax:
            return

        ex, ey = event.x, event.y

        min_dist = np.inf
        min_idx = -1
        for idx in range(len(x)):
            x3d, y3d, z3d = x[idx], y[idx], z[idx]
            x2d, y2d, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
            xdisp, ydisp = ax.transData.transform((x2d, y2d))
            dist = np.hypot(xdisp - ex, ydisp - ey)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        if min_dist < 30:
            mask_path = names[min_idx]
            mask = load_mask(mask_path)
            if mask_transform:
                mask = mask_transform(mask)
            p = multiprocessing.Process(target=show_3d_mask, args=(mask, mask_path))
            p.start()
    
    fig.canvas.mpl_connect('button_press_event', on_scrool_wheel_click)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from clip_dataloader import clip3d_subjects_withName
    from model import CLIP
    from tqdm import tqdm
    import torchio as tio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_dir_val = "/data/Data/laa_measures/val/quantile_normalized.json"
    data_dir = "/data/Data/cropped_laa128_64mm/masks"

    # define data loader 
    ds_val = clip3d_subjects_withName(data_dir, json_dir_val)
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

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
            unpooled_txt_features = unpooled_txt_features / unpooled_txt_features.norm(dim=2, keepdim=True)
        
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

    # UPDATED CALL: 'clustering_type' replaces 'context_var'
    visualize_latent_space3D(z_mu_array,
                             context_vars,
                             name_list,
                             proj_type="pca",
                             proj_comps=[0,1,2],
                             clustering_type='kmeans:k=6')
    # Note: You can change 'clustering_type' to any other supported clustering method
    # For example:
    #'kmeans:k=6'
    #'agglomerative:n_clusters=7,linkage=ward'
    #'spectral:n_clusters=8,n_neighbors=12'
    #'dbscan:eps=0.4,min_samples=10'
    #'hdbscan:min_cluster_size=12'