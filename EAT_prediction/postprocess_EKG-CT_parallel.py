import os
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tools as tools

threshold = 5e6  # Global threshold for components

def process_file(file_name, input_folder, output_folder):
    if not file_name.endswith(".nii.gz"):
        return 0  # Skip non-NIfTI files
    
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # Load the NIfTI file
    img_sitk = tools.load_nifti_as_sitk(input_path)
    img = tools.convert_sitk_to_numpy(img_sitk)

    # Perform connected component analysis
    labeled_data, num_features = label(img)

    no_components_removed = 0

    if num_features > 1:  # More than one connected component
        no_components_removed = 1
        # Remove small blobs
        component_sizes = np.bincount(labeled_data.ravel())[1:]  # Exclude background
        for component_label, size in enumerate(component_sizes, start=1):
            if size < threshold:
                labeled_data[labeled_data == component_label] = 0

    # Save the processed image
    processed_img = (labeled_data > 0).astype(np.uint8)
    processed_img_sitk = tools.convert_numpy_to_sitk(processed_img)
    tools.save_sitk_as_nifti_from_ref(processed_img_sitk, img_sitk, output_path)

    return no_components_removed

def connected_component_analysis_parallel(input_folder, output_folder, n_workers=8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_names = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]
    total_removed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, f, input_folder, output_folder): f for f in file_names}

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            total_removed += future.result()

    print(f"Number of files with multiple connected components removed: {total_removed}")

# Example usage
input_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg"
output_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed"

connected_component_analysis_parallel(input_folder, output_folder, n_workers=8)