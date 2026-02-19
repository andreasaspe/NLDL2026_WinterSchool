import tools as tools
import os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

data_folder = "/storage/Data/DTU-CGPS-1/NIFTI/"
output_folder_root = "/storage/awias/NLDL_Winterschool/NIFTI_reoriented/"

EKG_path = "/storage/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"
data_df = pd.read_csv(EKG_path)
all_series = data_df['NIFTI'].tolist()

# Create output folder if it doesn't exist
os.makedirs(output_folder_root, exist_ok=True)

for series in tqdm(all_series):
    filepath = os.path.join(data_folder, series + ".nii.gz")
    # Load the file using SimpleITK
    file_sitk = sitk.ReadImage(filepath)
    # Reorient the image to LPS
    file_sitk = tools.reorient_sitk(file_sitk, "LPS")
    # Save the reoriented image
    output_filepath = os.path.join(output_folder_root, series + "_img.nii.gz")
    sitk.WriteImage(file_sitk, output_filepath)