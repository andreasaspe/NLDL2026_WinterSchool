"""
Visualize latent space of CLIP model trained on EAT masks + ECG features.

Features:
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Color by clinical variables
- Multi-modal alignment visualization
- Cosine similarity heatmaps
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
import torchio as tio
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/awias/Documents/code/NLDL2026_WinterSchool/3DCLIP')
from model import CLIP
from clip_dataloader import clip3d_ecg_dataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained CLIP model."""
    embed_dim = 128
    image_resolution = 192
    vision_layers = (3, 4, 6, 3)
    vision_width = 64
    context_length = 36
    transformer_width = 256
    transformer_heads = 4
    transformer_layers = 4

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width,
        context_length, transformer_width, transformer_heads, transformer_layers,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded model from {checkpoint_path}")
    else:
        print("⚠ No checkpoint loaded, using random weights")

    model.eval()
    return model


def extract_embeddings(model, dataloader, device='cuda', compute_volumes=True):
    """Extract EAT and ECG embeddings from all samples in dataloader."""
    eat_embeddings = []
    ecg_embeddings = []
    eat_volumes = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['mask'][tio.DATA].to(device)
            context = batch['context'].to(device)
            
            # Get normalized embeddings
            eat_emb = model.encode_image(images)
            ecg_emb = model.encode_text(context)
            
            # Normalize
            eat_emb = eat_emb / (eat_emb.norm(dim=1, keepdim=True) + 1e-8)
            ecg_emb = ecg_emb / (ecg_emb.norm(dim=1, keepdim=True) + 1e-8)
            
            eat_embeddings.append(eat_emb.cpu().numpy())
            ecg_embeddings.append(ecg_emb.cpu().numpy())
            
            # Compute EAT volumes (voxel count, can be converted to mm³ if spacing known)
            if compute_volumes:
                volumes = (images > 0).sum(dim=(1, 2, 3, 4)).cpu().numpy()
                eat_volumes.append(volumes)
    
    eat_embeddings = np.vstack(eat_embeddings)
    ecg_embeddings = np.vstack(ecg_embeddings)
    
    if compute_volumes:
        eat_volumes = np.concatenate(eat_volumes)
        print(f"✓ Extracted {len(eat_embeddings)} embeddings and volumes")
        print(f"  EAT shape: {eat_embeddings.shape}, ECG shape: {ecg_embeddings.shape}")
        print(f"  EAT volume range: {eat_volumes.min():.0f} - {eat_volumes.max():.0f} voxels")
        return eat_embeddings, ecg_embeddings, eat_volumes
    else:
        print(f"✓ Extracted {len(eat_embeddings)} embeddings")
        print(f"  EAT shape: {eat_embeddings.shape}, ECG shape: {ecg_embeddings.shape}")
        return eat_embeddings, ecg_embeddings, None


def reduce_dimensions(embeddings, method='umap', n_components=2, random_state=42):
    """Reduce embeddings to 2D or 3D."""
    print(f"Reducing dimensions with {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                       perplexity=min(30, len(embeddings)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state,
                           n_neighbors=min(15, len(embeddings)-1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    
    if method == 'pca':
        print(f"  Explained variance: {reducer.explained_variance_ratio_[:n_components].sum():.2%}")
    
    return reduced


def plot_embeddings_2d(reduced_emb, labels, title, cmap='viridis', figsize=(10, 8),
                        save_path=None, continuous=True):
    """Plot 2D scatter of embeddings colored by labels."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if continuous:
        scatter = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], 
                           c=labels, cmap=cmap, s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=title)
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(reduced_emb[mask, 0], reduced_emb[mask, 1],
                      label=str(label), s=20, alpha=0.7)
        ax.legend()
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_alignment(eat_reduced, ecg_reduced, cosine_scores=None, 
                   title="EAT-ECG Alignment", save_path=None, n_samples=100):
    """Plot paired embeddings with lines connecting EAT and ECG."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sample for visibility
    if len(eat_reduced) > n_samples:
        indices = np.random.choice(len(eat_reduced), n_samples, replace=False)
        eat_sample = eat_reduced[indices]
        ecg_sample = ecg_reduced[indices]
        if cosine_scores is not None:
            scores_sample = cosine_scores[indices]
    else:
        eat_sample = eat_reduced
        ecg_sample = ecg_reduced
        scores_sample = cosine_scores
    
    # Draw lines
    for i in range(len(eat_sample)):
        color = plt.cm.RdYlGn(scores_sample[i]) if scores_sample is not None else 'gray'
        alpha = 0.3 if scores_sample is not None else 0.2
        ax.plot([eat_sample[i, 0], ecg_sample[i, 0]], 
               [eat_sample[i, 1], ecg_sample[i, 1]], 
               color=color, alpha=alpha, linewidth=1)
    
    # Plot points
    ax.scatter(eat_sample[:, 0], eat_sample[:, 1], 
              c='blue', s=50, alpha=0.7, label='EAT', edgecolors='black', linewidth=0.5)
    ax.scatter(ecg_sample[:, 0], ecg_sample[:, 1], 
              c='red', s=50, alpha=0.7, label='ECG', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_cosine_similarity_heatmap(eat_embeddings, ecg_embeddings, 
                                   max_samples=100, save_path=None):
    """Plot cosine similarity heatmap between EAT and ECG embeddings."""
    # Sample for visibility
    if len(eat_embeddings) > max_samples:
        indices = np.random.choice(len(eat_embeddings), max_samples, replace=False)
        eat_sample = eat_embeddings[indices]
        ecg_sample = ecg_embeddings[indices]
    else:
        eat_sample = eat_embeddings
        ecg_sample = ecg_embeddings
    
    # Compute cosine similarity matrix
    similarity_matrix = eat_sample @ ecg_sample.T
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='RdYlGn', center=0, 
               vmin=-1, vmax=1, square=True, 
               xticklabels=False, yticklabels=False,
               cbar_kws={'label': 'Cosine Similarity'})
    
    ax.set_xlabel('ECG Embeddings')
    ax.set_ylabel('EAT Embeddings')
    ax.set_title(f'Cross-Modal Cosine Similarity Matrix\n({len(eat_sample)} samples)')
    
    # Add diagonal line
    ax.plot([0, len(eat_sample)], [0, len(eat_sample)], 
           'b--', linewidth=2, label='Diagonal (same patient)')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def main():
    # ======================== Configuration ========================

    checkpoint_folder_name = 'glorious-snowball-42' #'major-violet-9' #"easy-river-18"  # Update this to your actual folder name
    
    # Paths
    checkpoint_path = f"/data/awias/NLDL_Winterschool/models/{checkpoint_folder_name}/best_clip3d_ecg.pth"  # Update this!
    data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
    csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"
    output_dir = f"/data/awias/NLDL_Winterschool/latent_visualizations/{checkpoint_folder_name}"
    
    # Settings
    split = 'test'  # 'train', 'val', or 'test'
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reduction_method = 'tsne'  # 'pca', 'tsne', or 'umap'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ======================== Load Data ========================
    
    print(f"\n{'='*60}")
    print(f"Loading {split} dataset...")
    print(f"{'='*60}\n")
    
    dataset = clip3d_ecg_dataset(data_dir, csv_path, augment=False, split=split)
    dataloader = tio.SubjectsLoader(dataset, batch_size=batch_size, 
                                    num_workers=0, shuffle=False)
    
    # Load clinical metadata - filter to match dataset
    df = pd.read_csv(csv_path)
    df = df[df['split'] == split].reset_index(drop=True)
    
    # Apply same filtering as dataset: only keep rows with existing mask files
    mask_suffix = '_EAT.nii.gz'
    mask_files = set(os.listdir(data_dir))
    df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
    df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
    df = df.drop(columns=['mask_file'])  # Clean up temporary column
    
    print(f"✓ Loaded {len(df)} clinical records for {split} split (with existing masks)")
    print(f"  Dataset size: {len(dataset)} (should match!)")
    
    # ======================== Load Model & Extract Embeddings ========================
    
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}\n")
    model = load_model(checkpoint_path, device)
    
    # After loading model, before extracting embeddings:
    print(f"\n🔍 Model Diagnostics:")
    print(f"  Logit scale: {model.logit_scale.item():.3f} (exp: {model.logit_scale.exp().item():.1f})")
    print(f"  Expected range: 2.7 - 4.6 (corresponding to exp(2.7)=15 to exp(4.6)=100)")

    # Check if model is actually trained
    sample_weight = model.visual.conv1.weight[0, 0, 0, 0, :3].detach().cpu()
    print(f"  Sample conv1 weights: {sample_weight.numpy()}")
    print(f"  (Random init would be close to N(0, 0.02))")

    print("Extract embeddings")
    eat_embeddings, ecg_embeddings, eat_volumes = extract_embeddings(model, dataloader, device, compute_volumes=True)
    
    # Compute alignment metrics
    cosine_scores = (eat_embeddings * ecg_embeddings).sum(axis=1)
    print(f"\n✓ Alignment metrics:")
    print(f"  Mean cosine similarity: {cosine_scores.mean():.3f} ± {cosine_scores.std():.3f}")
    print(f"  Min: {cosine_scores.min():.3f}, Max: {cosine_scores.max():.3f}")
    
    # ======================== Dimensionality Reduction ========================
    
    print(f"\n{'='*60}")
    print("Dimensionality reduction...")
    print(f"{'='*60}\n")
    
    # Reduce EAT embeddings
    eat_reduced = reduce_dimensions(eat_embeddings, method=reduction_method, n_components=2)
    
    # Reduce ECG embeddings
    ecg_reduced = reduce_dimensions(ecg_embeddings, method=reduction_method, n_components=2)
    
    # Combined (concatenated) reduction
    combined_embeddings = np.concatenate([eat_embeddings, ecg_embeddings], axis=1)
    combined_reduced = reduce_dimensions(combined_embeddings, method=reduction_method, n_components=2)
    
    # ======================== Visualizations ========================
    
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}\n")
    
    # 1. EAT embeddings colored by clinical variables
    
    # EAT volume
    if eat_volumes is not None:
        plot_embeddings_2d(eat_reduced, eat_volumes,
                          f"EAT Embeddings - EAT Volume ({reduction_method.upper()})",
                          cmap='viridis',
                          save_path=os.path.join(output_dir, f'eat_{reduction_method}_volume.png'))
    
    if 'clin_sex' in df.columns:
        plot_embeddings_2d(eat_reduced, df['clin_sex'].values, 
                          f"EAT Embeddings - Sex ({reduction_method.upper()})",
                          cmap='coolwarm', continuous=False,
                          save_path=os.path.join(output_dir, f'eat_{reduction_method}_sex.png'))
    
    if 'clin_weight' in df.columns:
        plot_embeddings_2d(eat_reduced, df['clin_weight'].values,
                          f"EAT Embeddings - Weight ({reduction_method.upper()})",
                          save_path=os.path.join(output_dir, f'eat_{reduction_method}_weight.png'))
    
    if 'low_voltage' in df.columns:
        plot_embeddings_2d(eat_reduced, df['low_voltage'].astype(int).values,
                          f"EAT Embeddings - Low Voltage ECG ({reduction_method.upper()})",
                          cmap='RdYlGn', continuous=False,
                          save_path=os.path.join(output_dir, f'eat_{reduction_method}_low_voltage.png'))
    
    # 2. ECG embeddings colored by all variables
    
    # EAT volume
    if eat_volumes is not None:
        plot_embeddings_2d(ecg_reduced, eat_volumes,
                          f"ECG Embeddings - EAT Volume ({reduction_method.upper()})",
                          cmap='viridis',
                          save_path=os.path.join(output_dir, f'ecg_{reduction_method}_volume.png'))
    
    if 'clin_sex' in df.columns:
        plot_embeddings_2d(ecg_reduced, df['clin_sex'].values,
                          f"ECG Embeddings - Sex ({reduction_method.upper()})",
                          cmap='coolwarm', continuous=False,
                          save_path=os.path.join(output_dir, f'ecg_{reduction_method}_sex.png'))
    
    if 'clin_weight' in df.columns:
        plot_embeddings_2d(ecg_reduced, df['clin_weight'].values,
                          f"ECG Embeddings - Weight ({reduction_method.upper()})",
                          save_path=os.path.join(output_dir, f'ecg_{reduction_method}_weight.png'))
    
    if 'low_voltage' in df.columns:
        plot_embeddings_2d(ecg_reduced, df['low_voltage'].astype(int).values,
                          f"ECG Embeddings - Low Voltage ECG ({reduction_method.upper()})",
                          cmap='RdYlGn', continuous=False,
                          save_path=os.path.join(output_dir, f'ecg_{reduction_method}_low_voltage.png'))
    
    # Alignment score
    plot_embeddings_2d(ecg_reduced, cosine_scores,
                      f"ECG Embeddings - Alignment Score ({reduction_method.upper()})",
                      cmap='RdYlGn',
                      save_path=os.path.join(output_dir, f'ecg_{reduction_method}_alignment.png'))
    
    # 3. Combined embeddings colored by all variables
    
    # EAT volume
    if eat_volumes is not None:
        plot_embeddings_2d(combined_reduced, eat_volumes,
                          f"Combined Embeddings - EAT Volume ({reduction_method.upper()})",
                          cmap='viridis',
                          save_path=os.path.join(output_dir, f'combined_{reduction_method}_volume.png'))
    
    if 'clin_sex' in df.columns:
        plot_embeddings_2d(combined_reduced, df['clin_sex'].values,
                          f"Combined Embeddings - Sex ({reduction_method.upper()})",
                          cmap='coolwarm', continuous=False,
                          save_path=os.path.join(output_dir, f'combined_{reduction_method}_sex.png'))
    
    if 'clin_weight' in df.columns:
        plot_embeddings_2d(combined_reduced, df['clin_weight'].values,
                          f"Combined Embeddings - Weight ({reduction_method.upper()})",
                          save_path=os.path.join(output_dir, f'combined_{reduction_method}_weight.png'))
    
    if 'low_voltage' in df.columns:
        plot_embeddings_2d(combined_reduced, df['low_voltage'].astype(int).values,
                          f"Combined Embeddings - Low Voltage ECG ({reduction_method.upper()})",
                          cmap='RdYlGn', continuous=False,
                          save_path=os.path.join(output_dir, f'combined_{reduction_method}_low_voltage.png'))
    
    # Alignment score
    plot_embeddings_2d(combined_reduced, cosine_scores,
                      f"Combined Embeddings - Alignment Score ({reduction_method.upper()})",
                      cmap='RdYlGn',
                      save_path=os.path.join(output_dir, f'combined_{reduction_method}_alignment.png'))
    
    # 4. Multi-modal alignment visualization
    plot_alignment(eat_reduced, ecg_reduced, cosine_scores,
                  title=f"EAT-ECG Alignment ({reduction_method.upper()})",
                  save_path=os.path.join(output_dir, f'alignment_{reduction_method}.png'),
                  n_samples=min(200, len(eat_reduced)))
    
    # 5. Cosine similarity heatmap
    plot_cosine_similarity_heatmap(eat_embeddings, ecg_embeddings,
                                   max_samples=100,
                                   save_path=os.path.join(output_dir, 'cosine_similarity_heatmap.png'))
    
    # ======================== Statistics ========================
    
    print(f"\n{'='*60}")
    print("Statistics Summary")
    print(f"{'='*60}\n")
    
    # EAT volume statistics
    if eat_volumes is not None:
        print(f"EAT Volume Statistics:")
        print(f"  Mean: {eat_volumes.mean():.0f} voxels")
        print(f"  Median: {np.median(eat_volumes):.0f} voxels")
        print(f"  Std: {eat_volumes.std():.0f} voxels")
        print(f"  Range: {eat_volumes.min():.0f} - {eat_volumes.max():.0f} voxels")
        
        # Correlation with alignment
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(eat_volumes, cosine_scores)
        spearman_r, spearman_p = spearmanr(eat_volumes, cosine_scores)
        print(f"\nCorrelation between EAT volume and alignment score:")
        print(f"  Pearson r={pearson_r:.3f}, p={pearson_p:.4f}")
        print(f"  Spearman r={spearman_r:.3f}, p={spearman_p:.4f}")
    
    # Alignment by clinical variables
    if 'low_voltage' in df.columns:
        low_voltage_mask = df['low_voltage'].astype(bool).values
        print(f"Alignment by low-voltage ECG:")
        print(f"  Low voltage (n={low_voltage_mask.sum()}): {cosine_scores[low_voltage_mask].mean():.3f}")
        print(f"  Normal (n={(~low_voltage_mask).sum()}): {cosine_scores[~low_voltage_mask].mean():.3f}")
    
    if 'clin_sex' in df.columns:
        print(f"\nAlignment by sex:")
        for sex in df['clin_sex'].unique():
            mask = df['clin_sex'].values == sex
            print(f"  Sex={sex} (n={mask.sum()}): {cosine_scores[mask].mean():.3f}")

    
    # Save embeddings for further analysis
    save_embeddings_path = os.path.join(output_dir, f'{split}_embeddings.npz')
    np.savez(save_embeddings_path,
             eat_embeddings=eat_embeddings,
             ecg_embeddings=ecg_embeddings,
             eat_reduced=eat_reduced,
             ecg_reduced=ecg_reduced,
             combined_reduced=combined_reduced,
             cosine_scores=cosine_scores,
             eat_volumes=eat_volumes)
    print(f"\n✓ Saved embeddings to: {save_embeddings_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == "__main__":
    main()
