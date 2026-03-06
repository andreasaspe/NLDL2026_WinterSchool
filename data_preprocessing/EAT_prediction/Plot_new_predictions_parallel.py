import os
import tools as tools
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

root = "/storage/awias/NLDL_Winterschool"
outputroot = "/storage/awias/NLDL_Winterschool/predictions_periseg_reoriented_plots"

os.makedirs(outputroot, exist_ok=True)

path_img = os.path.join(root, "NIFTI_reoriented")
path_pred = os.path.join(root, "predictions_periseg_reoriented")


def process_series(series):
    try:
        img_file = f"{series}_img.nii.gz"
        pred_file = f"{series}_pred.nii.gz"

        img_path = os.path.join(path_img, img_file)
        pred_path = os.path.join(path_pred, pred_file)

        img_sitk = sitk.ReadImage(img_path)
        pred_sitk = sitk.ReadImage(pred_path)

        img = sitk.GetArrayFromImage(img_sitk)
        pred = sitk.GetArrayFromImage(pred_sitk)

        spacing = img_sitk.GetSpacing()[::-1]

        output_path = os.path.join(outputroot, f"{series}.png")

        tools.plot_central_slice_img_mask_zyx(
            img,
            pred,
            spacing,
            title=series,
            output_path=output_path
        )

        return series

    except Exception as e:
        return f"ERROR {series}: {e}"


def main():

    all_series = [
        "_".join(x.split("_")[:3])
        for x in os.listdir(path_pred)
        if x.endswith(".nii.gz")
    ]

    workers = max(os.cpu_count() - 1, 1)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(executor.map(process_series, all_series), total=len(all_series)):
            pass


if __name__ == "__main__":
    main()