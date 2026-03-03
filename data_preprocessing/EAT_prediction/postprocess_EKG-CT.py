import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tools as tools
from tqdm import tqdm

def connected_component_analysis(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    largest_connected_components = []
    second_largest_connected_components = []
    
    threshold = 5e6

    no_components_removed = 0

    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith(".nii.gz"):
            
            # if not file_name.startswith("Pericardium-1_0038_SERIES0017"):
            #     continue

            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Load the NIfTI file
            img_sitk = tools.load_nifti_as_sitk(input_path)
            img = tools.convert_sitk_to_numpy(img_sitk)

            # Perform connected component analysis
            labeled_data, num_features = label(img)
            
            # Calculate sizes of connected components
            component_sizes = np.bincount(labeled_data.ravel())[1:]  # Exclude background (label 0)
            
            if num_features > 1: # More than one connected component - remove blobs.
                print("Found more than one connected component in file:", file_name)
                no_components_removed += 1
                # Remove all blobs smaller than the threshold
                for component_label, size in enumerate(component_sizes, start=1):
                    if size < threshold:
                        labeled_data[labeled_data == component_label] = 0

            # Save the processed image
            processed_img = (labeled_data > 0).astype(np.uint8)
                    
            # Save the processed image
            processed_img_sitk = tools.convert_numpy_to_sitk(processed_img)
            tools.save_sitk_as_nifti_from_ref(processed_img_sitk, img_sitk, output_path)
            
    print(f"Number of files with multiple connected components removed: {no_components_removed}")
    return

# Example usage
input_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"

# Perform connected component analysis
connected_component_analysis(input_folder, output_folder)

