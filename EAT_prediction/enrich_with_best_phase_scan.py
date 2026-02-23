import pandas as pd

best_phase_scan_path = "/storage/Data/DTU-CGPS-1/Filelist/DTU-CGPS-1_all_cardiac_ED_CE_clean_version_2.csv"
data_path = "/storage/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized.xlsx"
output_path = "/storage/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"

# Load the best phase scan data
best_phase_df = pd.read_csv(best_phase_scan_path)
# Load the pseudo_id translation data
data_df = pd.read_excel(data_path)

# Merge the best phase scan data with the pseudo_id translation data
# pseudonymized_id	filename	patient_id	series

# PseudoID	NIFTI

filtered_best_phase_df = best_phase_df[
    best_phase_df['NIFTI'].str.startswith('CGPS', na=False)
]

# remove rows where NIFTI is empty

data_df = data_df.merge(
    filtered_best_phase_df[['PseudoID', 'NIFTI']],
    left_on='pseudo_id',
    right_on='PseudoID',
    how='left')

# Put the NIFTI column right after the pseudo_id column
cols = (
    [data_df.columns[0]] +          # keep first column
    ['NIFTI'] +                     # place NIFTI second
    [col for col in data_df.columns if col not in [data_df.columns[0], 'NIFTI']]
)
data_df = data_df[cols]

# remove rows where NIFTI is empty (don't know why it happens)
data_df = data_df[data_df['NIFTI'].notna()]

# Save the enriched data to a new CSV file
data_df.to_csv(output_path, index=False)