import numpy as np
import os
root = "/data/awias/NLDL_Winterschool/latent_visualizations/glorious-snowball-42"
d = np.load(os.path.join(root, 'test_embeddings.npz'))
eat = d['eat_embeddings']
ecg = d['ecg_embeddings']

# Check embedding collapse - are all embeddings roughly the same?
print('=== Embedding diversity ===')
print(f'EAT embedding std across samples (per dim): {eat.std(axis=0).mean():.4f}')
print(f'ECG embedding std across samples (per dim): {ecg.std(axis=0).mean():.4f}')

# Cosine sim between random EAT pairs
n = min(500, len(eat))
eat_self_sim = []
ecg_self_sim = []
cross_sim = []
for _ in range(5000):
    i, j = np.random.choice(n, 2, replace=False)
    eat_self_sim.append(eat[i] @ eat[j])
    ecg_self_sim.append(ecg[i] @ ecg[j])
    cross_sim.append(eat[i] @ ecg[j])

print(f'\nEAT-EAT cosine (random pairs): {np.mean(eat_self_sim):.4f} ± {np.std(eat_self_sim):.4f}')
print(f'ECG-ECG cosine (random pairs): {np.mean(ecg_self_sim):.4f} ± {np.std(ecg_self_sim):.4f}')
print(f'EAT-ECG cosine (random pairs): {np.mean(cross_sim):.4f} ± {np.std(cross_sim):.4f}')

# Matched vs unmatched
cs = d['cosine_scores']  # diagonal = matched pairs
print(f'\nMatched EAT-ECG cosine: {cs.mean():.4f} ± {cs.std():.4f}')
print(f'Gap (matched - random cross): {cs.mean() - np.mean(cross_sim):.4f}')

# Check if EAT embeddings cluster by volume
ev = d['eat_volumes']
from scipy.stats import spearmanr
# correlation of EAT volume with first few PCA components
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
eat_pca = pca.fit_transform(eat)
print(f'\nEAT volume correlation with PCA components:')
for i in range(5):
    r, p = spearmanr(ev, eat_pca[:, i])
    print(f'  PC{i+1} (var={pca.explained_variance_ratio_[i]:.3f}): r={r:.3f}, p={p:.2e}')