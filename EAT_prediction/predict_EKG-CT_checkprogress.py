import SimpleITK as sitk
import numpy as np
from skimage.measure import label
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd

model_folder = "/data/awias/nnUNet/nnUNet_results/Dataset001_Periseg/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/"
input_data_folder = "/storage/Data/DTU-CGPS-1/NIFTI/"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg"
EKG_path = "/storage/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"

data_df = pd.read_csv(EKG_path)
all_series = data_df['NIFTI'].tolist()

print("Predict with NN U-Net")
os.environ["nnUNet_results"] = "/data/awias/nnUNet/nnUNet_results/"

in_files = []
out_files = []

for series in all_series:
    in_files.append([os.path.join(input_data_folder, series + ".nii.gz")])
    out_files.append(os.path.join(output_folder, series + "_pred.nii.gz"))

existing_outfiles = [f for f in out_files if os.path.exists(f)]
print(f"{len(existing_outfiles)} outfiles exist out of {len(out_files)} total.")