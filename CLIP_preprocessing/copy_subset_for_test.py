import os
import shutil

# Source and destination folders
src_folder = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
dst_folder = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm_test"

os.makedirs(dst_folder, exist_ok=True)

# Copy every 5th file from the source to the destination
def copy_subset(src, dst, step=5):
    files = sorted([f for f in os.listdir(src) if f.endswith('.nii.gz')])
    subset = files[::step]
    print(f"Copying {len(subset)} of {len(files)} files...")
    for f in subset:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("Done.")

if __name__ == "__main__":
    copy_subset(src_folder, dst_folder, step=5)
