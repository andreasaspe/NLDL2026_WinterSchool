import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

label_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"
img_folder = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"

os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(label_folder)):
    if filename.endswith(".nii.gz"):
        series = filename.replace("_pred.nii.gz", "")
        label_path = os.path.join(label_folder, filename)
        img_path = os.path.join(img_folder, series + ".nii.gz")
        output_path = os.path.join(output_folder, series + "_EAT.nii.gz")

        # EAT is defined as the values -190 and 0 in the original image in the pericardial mask
        
        # Load the label and image
        label_sitk = sitk.ReadImage(label_path)
        img_sitk = sitk.ReadImage(img_path)

        # Convert to numpy arrays
        label = sitk.GetArrayFromImage(label_sitk)
        img = sitk.GetArrayFromImage(img_sitk)

        # Create a mask for EAT
        eat_mask = (label == 1) & (img >= -190) & (img <= -30)
        eat_mask = eat_mask.astype(np.uint8)

        # Create a new SimpleITK image for the EAT mask
        eat_mask_sitk = sitk.GetImageFromArray(eat_mask)
        eat_mask_sitk.CopyInformation(label_sitk)

        # Write the EAT mask to the output folder
        sitk.WriteImage(eat_mask_sitk, output_path)
        print(f"Processed {filename}, saved EAT mask to {output_path}")