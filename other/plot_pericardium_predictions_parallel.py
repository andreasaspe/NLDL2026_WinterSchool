import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import tools

predictions_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"
img_folder = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_plots"

os.makedirs(output_folder, exist_ok=True)


def process_file(pred_file):
    if not pred_file.endswith(".nii.gz"):
        return

    series = pred_file.replace("_pred.nii.gz", "")

    pred_path = os.path.join(predictions_folder, pred_file)
    img_path_full = os.path.join(img_folder, series + ".nii.gz")
    output_path = os.path.join(output_folder, f"{series}.png")

    # Read prediction
    pred_sitk = sitk.ReadImage(pred_path)
    pred = sitk.GetArrayFromImage(pred_sitk)

    # Read image
    img_sitk = sitk.ReadImage(img_path_full)
    img = sitk.GetArrayFromImage(img_sitk)

    spacing = img_sitk.GetSpacing()[::-1]

    tools.plot_central_slice_img_mask_zyx(
        img, pred, spacing,
        title=series,
        output_path=output_path
    )


if __name__ == "__main__":

    files = [f for f in os.listdir(predictions_folder) if f.endswith(".nii.gz")]

    # Use all available cores
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))