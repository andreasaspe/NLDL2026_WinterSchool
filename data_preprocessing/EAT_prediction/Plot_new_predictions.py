import numpy
import os
import tools as tools
import SimpleITK as sitk
from collections import defaultdict

root = "/storage/awias/NLDL_Winterschool"
outputroot= "/storage/awias/NLDL_Winterschool/predictions_periseg_reoriented_plots"

os.makedirs(outputroot, exist_ok=True)

path_img =os.path.join(root, "NIFTI_reoriented")
path_pred = os.path.join(root, "predictions_periseg_reoriented")
all_series = ["_".join(x.split("_")[:3]) for x in os.listdir(path_pred) if x.endswith(".nii.gz")]

# all_test_subjects = [x.split(".")[0] for x in all_filenames_pred]
for series in all_series:
    img_file = f"{series}_img.nii.gz"
    pred_file = f"{series}_pred.nii.gz"

    img_path = os.path.join(path_img, img_file)
    pred_path = os.path.join(path_pred, pred_file)

    img_sitk = sitk.ReadImage(img_path)
    pred_sitk = sitk.ReadImage(pred_path)

    img = sitk.GetArrayFromImage(img_sitk)
    pred = sitk.GetArrayFromImage(pred_sitk)

    
    spacing = img_sitk.GetSpacing()[::-1]  # Reverse spacing to match ZYX order

    output_path = os.path.join(outputroot, f"{series}.png")

    tools.plot_central_slice_img_mask_zyx(img, pred, spacing, title=series, output_path=output_path)
    # tools.plot_central_slice_img_mask_zyx(img, pred, spacing, title=subject)
