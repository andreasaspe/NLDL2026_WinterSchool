"""
Analyze the trade-off between resampling spacing, network input size,
and the number of scans retained (i.e. those that fit within the target cube).

Reads all masks from the raw prediction folder, crops to tight bounding box,
then computes expected resampled size at various isotropic spacings.
"""

import SimpleITK as sitk
import numpy as np
import os
import tools

# ---- Settings ----
mask_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"
spacings = [0.5, 0.6, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0]
net_inputs = [128, 160, 192, 224, 256, 320, 384, 512]
sample_n = 200  # Set to e.g. 200 for a quick estimate, None for all

# ---- Collect sizes ----
files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.nii.gz')])
if sample_n is not None:
    files = files[:sample_n]

print(f"Processing {len(files)} masks from:\n  {mask_folder}\n")

# Store per-spacing results: list of (sx, sy, sz) expected sizes
results = {s: [] for s in spacings}
errors = 0

for i, f in enumerate(files):
    if i % 100 == 0:
        print(f"  [{i}/{len(files)}]", flush=True)
    try:
        mask = sitk.ReadImage(os.path.join(mask_folder, f))
        cropped = tools.crop_to_tighest_mask(mask)
        orig_spacing = np.array(cropped.GetSpacing())
        orig_size = np.array(cropped.GetSize())
        for sp in spacings:
            new_size = np.ceil(orig_size * orig_spacing / sp).astype(int)
            results[sp].append(new_size)
    except Exception:
        errors += 1

total = len(files) - errors
print(f"\nDone. Successfully processed {total} / {len(files)} masks ({errors} errors).\n")

# ---- Build table ----
header = f"{'Spacing':>8} | {'Net Input':>10} | {'Scans Kept':>12} | {'% Kept':>8} | {'Median Max Dim':>15}"
sep = "-" * len(header)

print("=" * len(header))
print("RESOLUTION vs DATA TRADE-OFF TABLE")
print("=" * len(header))
print(header)
print(sep)

for sp in spacings:
    sizes = np.array(results[sp])
    max_dims = sizes.max(axis=1)
    median_max = int(np.median(max_dims))

    for target in net_inputs:
        fits = np.all(sizes <= target, axis=1).sum()
        pct = 100 * fits / len(sizes)
        if pct == 0:
            continue  # skip rows where nothing fits
        print(f"{sp:>7.2f}mm | {target:>7d}^3 | {fits:>5d}/{len(sizes):<5d} | {pct:>7.1f}% | {median_max:>15d}")
    print(sep)

# ---- Summary of sweet-spot combos ----
print("\n\nSWEET-SPOT COMBOS (>=90% scans kept, input divisible by 32):")
print(f"{'Spacing':>8} | {'Net Input':>10} | {'Scans Kept':>12} | {'% Kept':>8}")
print("-" * 50)
for sp in spacings:
    sizes = np.array(results[sp])
    for target in net_inputs:
        if target % 32 != 0:
            continue
        fits = np.all(sizes <= target, axis=1).sum()
        pct = 100 * fits / len(sizes)
        if pct >= 90:
            print(f"{sp:>7.2f}mm | {target:>7d}^3 | {fits:>5d}/{len(sizes):<5d} | {pct:>7.1f}%")
            break  # show smallest fitting net input per spacing
