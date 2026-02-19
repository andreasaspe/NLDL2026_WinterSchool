import pandas as pd



csv_path = "/data/awias/ecg-ct/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"

df = pd.read_csv(csv_path)

print(df.columns.tolist())

ecg_columns = ['R_PeakAmpl_I', 'R_PeakAmpl_II', 'R_PeakAmpl_III', 'R_PeakAmpl_aVF', 'R_PeakAmpl_aVR', 'R_PeakAmpl_aVL', 'R_PeakAmpl_V1', 'R_PeakAmpl_V2', 'R_PeakAmpl_V3', 'R_PeakAmpl_V4', 'R_PeakAmpl_V5', 'R_PeakAmpl_V6', 'Q_PeakAmpl_I', 'Q_PeakAmpl_II', 'Q_PeakAmpl_III', 'Q_PeakAmpl_aVF', 'Q_PeakAmpl_aVR', 'Q_PeakAmpl_aVL', 'Q_PeakAmpl_V1', 'Q_PeakAmpl_V2', 'Q_PeakAmpl_V3', 'Q_PeakAmpl_V4', 'Q_PeakAmpl_V5', 'Q_PeakAmpl_V6', 'S_PeakAmpl_I', 'S_PeakAmpl_II', 'S_PeakAmpl_III', 'S_PeakAmpl_aVF', 'S_PeakAmpl_aVR', 'S_PeakAmpl_aVL', 'S_PeakAmpl_V1', 'S_PeakAmpl_V2', 'S_PeakAmpl_V3', 'S_PeakAmpl_V4', 'S_PeakAmpl_V5', 'S_PeakAmpl_V6', 'AtrialRate', 'VentricularRate']

ecg_data = df[ecg_columns]

print(ecg_data.head())