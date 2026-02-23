import SimpleITK as sitk
import numpy as np
import os
import tools as tools

mask_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"
img_folder = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"

os.makedirs(output_folder, exist_ok=True)

target_spacing = 0.5
target_size = (256, 256, 256)

too_large_count = 0
total_count = 0

for filename in os.listdir(mask_folder):
    print(f"{too_large_count}/{total_count} masks exceeded target size.")
    total_count += 1

    series = filename.replace("_EAT.nii.gz", "")
    mask_path = os.path.join(mask_folder, filename)
    img_path = os.path.join(img_folder, series + ".nii.gz")
    output_path = os.path.join(output_folder, series + "_EAT.nii.gz")

    # Load mask
    mask_sitk = sitk.ReadImage(mask_path)
    
    # Get size
    # size = mask_sitk.GetSize()
    # print(f"Original size: {size}")

    mask_cropped_sitk = tools.crop_to_tighest_mask(mask_sitk)
    
    # Get size
    # size = mask_cropped_sitk.GetSize()
    # print(f"Cropped size: {size}")

    # Resample mask
    mask_resampled_sitk = tools.resample_to_isotropic_spacing(mask_cropped_sitk, spacing = target_spacing, interpolation='nearest')

    # Get size
    # size = mask_resampled_sitk.GetSize()
    # print(f"Resampled size: {size}")

    size = mask_resampled_sitk.GetSize()
    if size[0] > target_size[0] or size[1] > target_size[1] or size[2] > target_size[2]:
        # print(f"Warning: Resampled mask size {size} exceeds target size {target_size}. Consider adjusting the cropping or resampling parameters.")
        too_large_count += 1
        continue

    # Pad mask to 256x256x256
    mask_padded_sitk = tools.pad_to_shape(mask_resampled_sitk,target_size=target_size,pad_value=0)

    # Save processed mask
    sitk.WriteImage(mask_padded_sitk, output_path)