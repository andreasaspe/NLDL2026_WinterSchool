e output folder if it doesn't exist
# os.makedirs(output_folder_root, exist_ok=True)


# for root, dirs, files in os.walk(data_folder):
#     for file in tqdm(files):
#         if file.endswith(".nii.gz"):
#             filepath = os.path.join(root, file)
#             # Load the file using SimpleITK
#             file_sitk = sitk.ReadImage(filepath)
#             # Reorient the image to LPS
#             file_sitk = tools.reorient_sitk(file_sitk, "LPS")
#             # Save the reoriented image
#             relative_path = os.path.relpath(filepath, data_folder)
#             output_filepath = os.path.join(output_folder_root, relative_path)
#             os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
#             sitk.WriteImage(file_sitk, output_filepath)