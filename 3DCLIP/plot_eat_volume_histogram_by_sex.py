"""
Plot overlapping histograms of EAT volume for men and women.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================== Configuration ========================

checkpoint_folder_name = 'glorious-snowball-42'
data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"
embeddings_path = f"/data/awias/NLDL_Winterschool/latent_visualizations/{checkpoint_folder_name}/test_embeddings.npz"
output_dir = f"/data/awias/NLDL_Winterschool/latent_visualizations/{checkpoint_folder_name}"
split = 'test'

os.makedirs(output_dir, exist_ok=True)

# ======================== Load Data ========================

data = np.load(embeddings_path)
eat_volumes = data['eat_volumes']  # in mL

df = pd.read_csv(csv_path)
df = df[df['split'] == split].reset_index(drop=True)

mask_suffix = '_EAT.nii.gz'
mask_files = set(os.listdir(data_dir))
df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
df = df.drop(columns=['mask_file'])

sex = df['clin_sex'].values
men_volumes = eat_volumes[sex == 1]
women_volumes = eat_volumes[sex == 0]

print(f"Men:   n={len(men_volumes)}, mean={men_volumes.mean():.1f} mL, std={men_volumes.std():.1f} mL")
print(f"Women: n={len(women_volumes)}, mean={women_volumes.mean():.1f} mL, std={women_volumes.std():.1f} mL")

# ======================== Plot ========================

plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(min(eat_volumes.min(), 0), eat_volumes.max(), 40)

ax.hist(men_volumes, bins=bins, alpha=0.6, label=f'Men (n={len(men_volumes)})', color='#2196F3', edgecolor='white', linewidth=0.5)
ax.hist(women_volumes, bins=bins, alpha=0.6, label=f'Women (n={len(women_volumes)})', color='#F44336', edgecolor='white', linewidth=0.5)

ax.set_xlabel('EAT Volume [mL]')
ax.set_ylabel('Count')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
save_path = os.path.join(output_dir, 'eat_volume_histogram_by_sex.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

plt.show()
