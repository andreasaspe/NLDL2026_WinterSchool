"""
Find best threshold to distinguish men vs women from:
1. t-SNE of EAT embeddings (best 1D threshold on either component)
2. EAT volume histogram

Reports accuracy for each.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

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
eat_embeddings = data['eat_embeddings']  # (N, 128)
eat_volumes = data['eat_volumes']        # (N,) in mL

df = pd.read_csv(csv_path)
df = df[df['split'] == split].reset_index(drop=True)
mask_suffix = '_EAT.nii.gz'
mask_files = set(os.listdir(data_dir))
df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
df = df.drop(columns=['mask_file'])

sex = df['clin_sex'].values  # 1=Men, 0=Women


def find_best_threshold(values, labels):
    """
    Sweep all midpoints between sorted unique values to find the threshold
    that maximizes accuracy (considering both polarities).
    Returns (best_threshold, best_accuracy, best_polarity).
    Polarity: 'gt' means predict 1 if value > threshold, 'lt' means predict 1 if value < threshold.
    """
    sorted_vals = np.sort(np.unique(values))
    # Candidate thresholds: midpoints between consecutive sorted values
    thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

    best_acc = 0.0
    best_thresh = 0.0
    best_pol = 'gt'

    for t in thresholds:
        pred_gt = (values > t).astype(int)
        pred_lt = (values < t).astype(int)
        acc_gt = accuracy_score(labels, pred_gt)
        acc_lt = accuracy_score(labels, pred_lt)
        if acc_gt > best_acc:
            best_acc = acc_gt
            best_thresh = t
            best_pol = 'gt'
        if acc_lt > best_acc:
            best_acc = acc_lt
            best_thresh = t
            best_pol = 'lt'

    return best_thresh, best_acc, best_pol


# ======================== 1. t-SNE Embedding Threshold ========================

print("Running t-SNE on EAT embeddings...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
eat_tsne = tsne.fit_transform(eat_embeddings)

# Try both t-SNE components
results_tsne = []
for comp in [0, 1]:
    thresh, acc, pol = find_best_threshold(eat_tsne[:, comp], sex)
    results_tsne.append((comp, thresh, acc, pol))
    print(f"  t-SNE Component {comp+1}: threshold={thresh:.2f}, accuracy={acc:.3f} ({100*acc:.1f}%), polarity={pol}")

# Pick best component
best_comp, best_thresh_tsne, best_acc_tsne, best_pol_tsne = max(results_tsne, key=lambda x: x[2])
print(f"\n→ Best t-SNE threshold: Component {best_comp+1}, "
      f"threshold={best_thresh_tsne:.2f}, accuracy={best_acc_tsne:.3f} ({100*best_acc_tsne:.1f}%)")

# ======================== 2. EAT Volume Threshold ========================

print("\nFinding best EAT volume threshold...")
thresh_vol, acc_vol, pol_vol = find_best_threshold(eat_volumes, sex)
print(f"→ Best EAT volume threshold: {thresh_vol:.1f} mL, "
      f"accuracy={acc_vol:.3f} ({100*acc_vol:.1f}%), polarity={pol_vol}")

# ======================== Plots ========================

plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

# --- Plot 1: t-SNE with threshold line ---
fig, ax = plt.subplots(figsize=(8, 7))
men_mask = sex == 1
ax.scatter(eat_tsne[men_mask, 0], eat_tsne[men_mask, 1],
           s=15, alpha=0.6, label='Men', c='#2196F3')
ax.scatter(eat_tsne[~men_mask, 0], eat_tsne[~men_mask, 1],
           s=15, alpha=0.6, label='Women', c='#F44336')

# Draw threshold line
if best_comp == 0:
    ax.axvline(best_thresh_tsne, color='black', linestyle='--', linewidth=2,
               label=f'Threshold (acc={100*best_acc_tsne:.1f}%)')
else:
    ax.axhline(best_thresh_tsne, color='black', linestyle='--', linewidth=2,
               label=f'Threshold (acc={100*best_acc_tsne:.1f}%)')

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_path = os.path.join(output_dir, 'tsne_sex_threshold.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

# --- Plot 2: EAT volume histogram with threshold ---
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(eat_volumes.min(), eat_volumes.max(), 40)
ax.hist(eat_volumes[men_mask], bins=bins, alpha=0.6, label='Men', color='#2196F3', edgecolor='white', linewidth=0.5)
ax.hist(eat_volumes[~men_mask], bins=bins, alpha=0.6, label='Women', color='#F44336', edgecolor='white', linewidth=0.5)
ax.axvline(thresh_vol, color='black', linestyle='--', linewidth=2,
           label=f'Threshold={thresh_vol:.0f} mL (acc={100*acc_vol:.1f}%)')
ax.set_xlabel('EAT Volume [mL]')
ax.set_ylabel('Count')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_path = os.path.join(output_dir, 'eat_volume_sex_threshold.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

# ======================== Summary ========================

print(f"\n{'='*60}")
print("SUMMARY: Sex Classification Accuracy")
print(f"{'='*60}")
print(f"  t-SNE (Component {best_comp+1}):  {100*best_acc_tsne:.1f}%  (threshold={best_thresh_tsne:.2f})")
print(f"  EAT Volume:              {100*acc_vol:.1f}%  (threshold={thresh_vol:.1f} mL)")
print(f"  Chance level:            {100*max(sex.mean(), 1-sex.mean()):.1f}%")
print(f"{'='*60}")

plt.show()
