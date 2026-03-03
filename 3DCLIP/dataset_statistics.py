"""
Print dataset statistics: split sizes, sex distribution, low-voltage counts, and acquisition time range.
"""

import os
import pandas as pd

csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"
data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"

df = pd.read_csv(csv_path)

# Filter to rows with existing mask files (same as training pipeline)
mask_suffix = '_EAT.nii.gz'
mask_files = set(os.listdir(data_dir))
df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
df = df.drop(columns=['mask_file'])

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

# Total
print(f"\nTotal scans: {len(df)}")

# Split sizes
print(f"\nSplit sizes:")
for split in ['train', 'val', 'test']:
    n = (df['split'] == split).sum()
    print(f"  {split:>5}: {n:>5}  ({100*n/len(df):.1f}%)")

# Sex distribution
print(f"\nSex distribution (1=Men, 0=Women):")
for split_name in ['all', 'train', 'val', 'test']:
    subset = df if split_name == 'all' else df[df['split'] == split_name]
    n_men = (subset['clin_sex'] == 1).sum()
    n_women = (subset['clin_sex'] == 0).sum()
    n_unknown = subset['clin_sex'].isna().sum()
    print(f"  {split_name:>5}: Men={n_men}, Women={n_women}"
          + (f", Unknown={n_unknown}" if n_unknown > 0 else "")
          + f"  ({100*n_men/len(subset):.1f}% men)")

# Low voltage
print(f"\nLow-voltage ECG:")
for split_name in ['all', 'train', 'val', 'test']:
    subset = df if split_name == 'all' else df[df['split'] == split_name]
    n_lv = (subset['low_voltage'] == 1).sum()
    n_normal = (subset['low_voltage'] == 0).sum()
    n_na = subset['low_voltage'].isna().sum()
    print(f"  {split_name:>5}: Low-voltage={n_lv}, Normal={n_normal}"
          + (f", Unknown={n_na}" if n_na > 0 else "")
          + f"  ({100*n_lv/len(subset):.1f}% low-voltage)")

# Weight & height
print(f"\nWeight [kg]:")
for split_name in ['all', 'train', 'val', 'test']:
    subset = df if split_name == 'all' else df[df['split'] == split_name]
    w = subset['clin_weight'].dropna()
    print(f"  {split_name:>5}: mean={w.mean():.1f}, std={w.std():.1f}, "
          f"range=[{w.min():.0f}, {w.max():.0f}], n_missing={subset['clin_weight'].isna().sum()}")

if 'clin_height' in df.columns:
    print(f"\nHeight [cm]:")
    for split_name in ['all', 'train', 'val', 'test']:
        subset = df if split_name == 'all' else df[df['split'] == split_name]
        h = subset['clin_height'].dropna()
        print(f"  {split_name:>5}: mean={h.mean():.1f}, std={h.std():.1f}, "
              f"range=[{h.min():.0f}, {h.max():.0f}], n_missing={subset['clin_height'].isna().sum()}")

# Acquisition time range
print(f"\nAcquisition time range:")
df['ekg_datetime'] = pd.to_datetime(df['ekg_datetime'], errors='coerce')
for split_name in ['all', 'train', 'val', 'test']:
    subset = df if split_name == 'all' else df[df['split'] == split_name]
    dates = subset['ekg_datetime'].dropna()
    if len(dates) > 0:
        print(f"  {split_name:>5}: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
              f"  ({(dates.max() - dates.min()).days} days span, n_missing={subset['ekg_datetime'].isna().sum()})")
    else:
        print(f"  {split_name:>5}: no valid dates")

print(f"\n{'=' * 60}")
