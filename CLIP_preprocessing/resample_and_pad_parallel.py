import SimpleITK as sitk
import numpy as np
import os
import tools as tools
from concurrent.futures import ProcessPoolExecutor, as_completed

mask_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"
img_folder = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
os.makedirs(output_folder, exist_ok=True)

target_spacing = 0.5
target_size = (320, 320, 320)

files = [f for f in os.listdir(mask_folder) if f.endswith("_EAT.nii.gz")]

def process_mask(filename):
    series = filename.replace("_EAT.nii.gz", "")
    mask_path = os.path.join(mask_folder, filename)
    img_path = os.path.join(img_folder, series + ".nii.gz")
    output_path = os.path.join(output_folder, series + "_EAT.nii.gz")
    try:
        mask_sitk = sitk.ReadImage(mask_path)
        mask_cropped_sitk = tools.crop_to_tighest_mask(mask_sitk)
        mask_resampled_sitk = tools.resample_to_isotropic_spacing(mask_cropped_sitk, spacing=target_spacing, interpolation='nearest')
        size = mask_resampled_sitk.GetSize()
        if size[0] > target_size[0] or size[1] > target_size[1] or size[2] > target_size[2]:
            return (filename, False, f"Too large: {size}")
        mask_padded_sitk = tools.pad_to_shape(mask_resampled_sitk, target_size=target_size, pad_value=0)
        sitk.WriteImage(mask_padded_sitk, output_path)
        return (filename, True, "OK")
    except Exception as e:
        return (filename, False, str(e))

if __name__ == "__main__":
    total_count = 0
    too_large_count = 0
    failed_count = 0
    max_workers = os.cpu_count() or 4
    print(f"Parallelizing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_mask, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            filename, success, msg = future.result()
            total_count += 1
            if not success:
                if "Too large" in msg:
                    too_large_count += 1
                else:
                    failed_count += 1
            if i % 50 == 0:
                print(f"{too_large_count}/{total_count} masks exceeded target size. {failed_count} failed. Last: {filename} ({msg})")
    print(f"Done. {too_large_count} too large, {failed_count} failed, {total_count} processed.")
