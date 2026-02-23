import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

label_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"
img_folder = "/storage/Data/DTU-CGPS-1/NIFTI"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"

os.makedirs(output_folder, exist_ok=True)


def process_file(filename):
    if not filename.endswith(".nii.gz"):
        return None

    series = filename.replace("_pred.nii.gz", "")
    label_path = os.path.join(label_folder, filename)
    img_path = os.path.join(img_folder, series + ".nii.gz")
    output_path = os.path.join(output_folder, series + "_EAT.nii.gz")

    try:
        # Load images
        label_sitk = sitk.ReadImage(label_path)
        img_sitk = sitk.ReadImage(img_path)

        # Convert to numpy
        label = sitk.GetArrayFromImage(label_sitk)
        img = sitk.GetArrayFromImage(img_sitk)

        # Create EAT mask
        eat_mask = (label == 1) & (img >= -190) & (img <= 0)
        eat_mask = eat_mask.astype(np.uint8)

        # Convert back to SITK
        eat_mask_sitk = sitk.GetImageFromArray(eat_mask)
        eat_mask_sitk.CopyInformation(label_sitk)

        # Save
        sitk.WriteImage(eat_mask_sitk, output_path)

        # return f"Processed {filename}"

    except Exception as e:
        return f"Error processing {filename}: {e}"


if __name__ == "__main__":

    filenames = os.listdir(label_folder)

    # Adjust max_workers depending on your CPU
    max_workers = os.cpu_count()  # or e.g. 8

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f) for f in filenames]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)