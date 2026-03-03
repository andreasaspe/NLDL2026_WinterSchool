import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

EAT_folder = "/storage/awias/NLDL_Winterschool/predictions_periseg_postprocessed_EAT"

max_voxel_dims = [0, 0, 0]  # x, y, z in voxels

for filename in os.listdir(EAT_folder):
    if not filename.endswith(".nii.gz"):
        continue

    path = os.path.join(EAT_folder, filename)
    img_sitk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_sitk)
    spacing = img_sitk.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
    
    # Bounding box of nonzero mask
    nonzero = np.nonzero(img)
    z_min, y_min, x_min = np.min(nonzero[0]), np.min(nonzero[1]), np.min(nonzero[2])
    z_max, y_max, x_max = np.max(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[2])
    
    # Size in mm along each axis
    size_mm = [
        (x_max - x_min + 1) * spacing[0],
        (y_max - y_min + 1) * spacing[1],
        (z_max - z_min + 1) * spacing[2],
    ]
    
    # Convert to target spacing 0.5 mm
    size_voxels = [int(np.ceil(s / 0.5)) for s in size_mm]

    # Update maximum over all images
    max_voxel_dims = [max(max_voxel_dims[i], size_voxels[i]) for i in range(3)]

print("Estimated maximal 3D size at 0.5 mm spacing (X, Y, Z):", max_voxel_dims)