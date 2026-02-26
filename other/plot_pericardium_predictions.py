import numpy
import os
import tools as tools
import SimpleITK as sitk
from collections import defaultdict
from tqdm import tqdm

predictions_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"
img_path = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder= "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_plots"

os.makedirs(output_folder, exist_ok=True)

for pred_file in tqdm(os.listdir(predictions_folder)):
    if pred_file.endswith(".nii.gz"):
        series = pred_file.replace("_pred.nii.gz", "")
        print(f"Processing: {pred_file}")
        # Read the prediction
        pred_path = os.path.join(predictions_folder, pred_file)
        pred_sitk = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(pred_sitk)
        
        # Read the corresponding image
        img_file = series + ".nii.gz"
        img_path_full = os.path.join(img_path, img_file)
        img_sitk = sitk.ReadImage(img_path_full)
        img = sitk.GetArrayFromImage(img_sitk)
        
        # Plot and save
        output_path = os.path.join(output_folder, f"{series}.png")
        spacing = img_sitk.GetSpacing()[::-1]  # Reverse to get (z, y, x)
        tools.plot_central_slice_img_mask_zyx(img, pred, spacing, title=series, output_path=output_path)
