import tools as tools
import os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

data_folder = "/storage/Data/DTU-CGPS-1/NIFTI/"
output_folder_root = "/storage/awias/NLDL_Winterschool/NIFTI_reoriented/"

EKG_path = "/storage/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"
data_df = pd.read_csv(EKG_path)
all_series = data_df['NIFTI'].tolist()

# Create output folder if it doesn't exist
os.makedirs(output_folder_root, exist_ok=True)

def process_series(series):
    try:
        filepath = os.path.join(data_folder, series + ".nii.gz")
        file_sitk = sitk.ReadImage(filepath)
        file_sitk = tools.reorient_sitk(file_sitk, "LPS")
        output_filepath = os.path.join(output_folder_root, series + "_img.nii.gz")
        sitk.WriteImage(file_sitk, output_filepath)
        return series, "success"
    except Exception as e:
        return series, f"failed: {e}"

# Use ThreadPoolExecutor for I/O-bound tasks
max_workers = 16 # adjust based on CPU cores and I/O bandwidth
results = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_series, series): series for series in all_series}
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

# Optional: save processing results
results_df = pd.DataFrame(results, columns=["Series", "Status"])
results_df.to_csv(os.path.join(output_folder_root, "processing_log.csv"), index=False)
