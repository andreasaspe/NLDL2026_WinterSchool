"""
Plot EAT volume vs body weight scatter plot.
Uses precomputed embeddings (with EAT volumes) and clinical metadata.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ======================== Configuration ========================

checkpoint_folder_name = 'glorious-snowball-42'
data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"
embeddings_path = f"/data/awias/NLDL_Winterschool/latent_visualizations/{checkpoint_folder_name}/test_embeddings.npz"
output_dir = f"/data/awias/NLDL_Winterschool/latent_visualizations/{checkpoint_folder_name}"
split = 'test'

os.makedirs(output_dir, exist_ok=True)

# ======================== Load Data ========================

# Load precomputed EAT volumes
data = np.load(embeddings_path)
eat_volumes = data['eat_volumes']  # in mL

# Load clinical metadata (same filtering as visualize_latent_space_ecg.py)
df = pd.read_csv(csv_path)
df = df[df['split'] == split].reset_index(drop=True)

mask_suffix = '_EAT.nii.gz'
mask_files = set(os.listdir(data_dir))
df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
df = df.drop(columns=['mask_file'])

weights = df['clin_weight'].values

# Drop NaNs
valid = ~np.isnan(weights)
eat_volumes = eat_volumes[valid]
weights = weights[valid]

# ======================== Statistics ========================

r_pearson, p_pearson = pearsonr(weights, eat_volumes)
r_spearman, p_spearman = spearmanr(weights, eat_volumes)
print(f"Pearson  r={r_pearson:.3f}, p={p_pearson:.2e}")
print(f"Spearman r={r_spearman:.3f}, p={p_spearman:.2e}")

# ======================== Plot ========================

plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(weights, eat_volumes, s=20, alpha=0.6, edgecolors='none', c='#2196F3')

# Fit and draw regression line
z = np.polyfit(weights, eat_volumes, 1)
p = np.poly1d(z)
x_line = np.linspace(weights.min(), weights.max(), 100)
ax.plot(x_line, p(x_line), '--', color='#F44336', linewidth=2,
        label=f'r={r_pearson:.2f} (p={p_pearson:.1e})')

ax.set_xlabel('Weight [kg]')
ax.set_ylabel('EAT Volume [mL]')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
save_path = os.path.join(output_dir, 'eat_volume_vs_weight.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

plt.show()
