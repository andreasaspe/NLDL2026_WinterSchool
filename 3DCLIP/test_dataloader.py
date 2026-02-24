#!/usr/bin/env python3
"""Quick test to verify the dataloader works with split column."""

import sys
sys.path.insert(0, '/home/awias/Documents/code/NLDL2026_WinterSchool/3DCLIP')

from clip_dataloader import clip3d_ecg_dataset

data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"

print("Testing dataloader with split column...")
print("=" * 60)

# Test train split
print("\n1. Loading TRAIN split:")
ds_train = clip3d_ecg_dataset(data_dir, csv_path, augment=False, split='train')
print(f"   Train dataset size: {len(ds_train)}")

# Test val split
print("\n2. Loading VAL split:")
ds_val = clip3d_ecg_dataset(data_dir, csv_path, augment=False, split='val')
print(f"   Val dataset size: {len(ds_val)}")

# Test test split
print("\n3. Loading TEST split:")
ds_test = clip3d_ecg_dataset(data_dir, csv_path, augment=False, split='test')
print(f"   Test dataset size: {len(ds_test)}")

print("\n" + "=" * 60)
print("SUCCESS! All splits loaded correctly.")
print(f"Total samples: {len(ds_train) + len(ds_val) + len(ds_test)}")

# Test loading a sample
print("\n4. Testing sample loading:")
sample = ds_train[0]
print(f"   Sample keys: {sample.keys()}")
print(f"   Mask shape: {sample['mask'].data.shape}")
print(f"   Context shape: {sample['context'].shape}")
print(f"   Context values (first 5): {sample['context'][:5].tolist()}")

print("\nAll tests passed! ✓")
