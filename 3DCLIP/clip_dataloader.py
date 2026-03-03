import torch 
import numpy as np
import os

import SimpleITK as sitk
import torchio as tio


class clip3d_ecg_dataset(tio.SubjectsDataset):
    """Dataset that pairs 3D masks with ECG features from a CSV file."""

    EKG_KEYS = [
        'R_PeakAmpl_I', 'R_PeakAmpl_II', 'R_PeakAmpl_III',
        'R_PeakAmpl_aVF', 'R_PeakAmpl_aVR', 'R_PeakAmpl_aVL',
        'R_PeakAmpl_V1', 'R_PeakAmpl_V2', 'R_PeakAmpl_V3',
        'R_PeakAmpl_V4', 'R_PeakAmpl_V5', 'R_PeakAmpl_V6',
        'Q_PeakAmpl_I', 'Q_PeakAmpl_II', 'Q_PeakAmpl_III',
        'Q_PeakAmpl_aVF', 'Q_PeakAmpl_aVR', 'Q_PeakAmpl_aVL',
        'Q_PeakAmpl_V1', 'Q_PeakAmpl_V2', 'Q_PeakAmpl_V3',
        'Q_PeakAmpl_V4', 'Q_PeakAmpl_V5', 'Q_PeakAmpl_V6',
        'S_PeakAmpl_I', 'S_PeakAmpl_II', 'S_PeakAmpl_III',
        'S_PeakAmpl_aVF', 'S_PeakAmpl_aVR', 'S_PeakAmpl_aVL',
        'S_PeakAmpl_V1', 'S_PeakAmpl_V2', 'S_PeakAmpl_V3',
        'S_PeakAmpl_V4', 'S_PeakAmpl_V5', 'S_PeakAmpl_V6',
    ]

    def __init__(self, data_dir, csv_path, augment=False, split='train',
                 mask_suffix='_EAT.nii.gz'):
        """
        Args:
            data_dir: Directory containing mask files
            csv_path: Path to CSV with 'split' column (train/val/test)
            augment: Whether to apply data augmentation
            split: Which split to use ('train', 'val', or 'test')
            mask_suffix: Suffix to append to NIFTI column to get mask filename
        """
        import pandas as pd

        self.data_dir = data_dir
        self.keys = self.EKG_KEYS

        df = pd.read_csv(csv_path)

        # Filter by split column
        if 'split' not in df.columns:
            raise ValueError(f"CSV file must contain 'split' column. Found columns: {df.columns.tolist()}")
        
        df = df[df['split'] == split].reset_index(drop=True)
        print(f"Found {len(df)} subjects in '{split}' split from CSV.")

        # Only keep rows whose mask file exists on disk (just preventing file not found errors later on)
        mask_files = set(os.listdir(data_dir))
        df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
        df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
        print(f"Found {len(df)} subjects with both ECG data and mask files in '{split}' split.")

        # Build subjects list
        subjects = []
        for _, row in df.iterrows():
            mask_path = os.path.join(self.data_dir, row['mask_file'])
            context = np.array([row[k] if pd.notna(row[k]) else 0.0
                                for k in self.keys], dtype=np.float32)
            subject = tio.Subject(
                mask=tio.LabelMap(mask_path),
                context=torch.from_numpy(context),
            )
            subjects.append(subject)

        self.augment = augment
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(
                    scales=1,
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5,
                ),
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)



class clip3d_ecg_dataset_nosplit(tio.SubjectsDataset):
    """Dataset that pairs 3D masks with ECG features from a CSV file."""

    EKG_KEYS = [
        'R_PeakAmpl_I', 'R_PeakAmpl_II', 'R_PeakAmpl_III',
        'R_PeakAmpl_aVF', 'R_PeakAmpl_aVR', 'R_PeakAmpl_aVL',
        'R_PeakAmpl_V1', 'R_PeakAmpl_V2', 'R_PeakAmpl_V3',
        'R_PeakAmpl_V4', 'R_PeakAmpl_V5', 'R_PeakAmpl_V6',
        'Q_PeakAmpl_I', 'Q_PeakAmpl_II', 'Q_PeakAmpl_III',
        'Q_PeakAmpl_aVF', 'Q_PeakAmpl_aVR', 'Q_PeakAmpl_aVL',
        'Q_PeakAmpl_V1', 'Q_PeakAmpl_V2', 'Q_PeakAmpl_V3',
        'Q_PeakAmpl_V4', 'Q_PeakAmpl_V5', 'Q_PeakAmpl_V6',
        'S_PeakAmpl_I', 'S_PeakAmpl_II', 'S_PeakAmpl_III',
        'S_PeakAmpl_aVF', 'S_PeakAmpl_aVR', 'S_PeakAmpl_aVL',
        'S_PeakAmpl_V1', 'S_PeakAmpl_V2', 'S_PeakAmpl_V3',
        'S_PeakAmpl_V4', 'S_PeakAmpl_V5', 'S_PeakAmpl_V6',
    ]

    def __init__(self, data_dir, csv_path, augment=False, train=True,
                 val_frac=0.2, seed=42, mask_suffix='_EAT.nii.gz'):
        import pandas as pd

        self.data_dir = data_dir
        self.keys = self.EKG_KEYS

        df = pd.read_csv(csv_path)

        # Only keep rows whose mask file exists on disk
        mask_files = set(os.listdir(data_dir))
        df['mask_file'] = df['NIFTI'].apply(lambda x: x + mask_suffix)
        df = df[df['mask_file'].isin(mask_files)].reset_index(drop=True)
        print(f"Found {len(df)} subjects with both ECG data and mask files.")

        # Train / val split (deterministic)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_frac))
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_val   = df.iloc[split_idx:].reset_index(drop=True)
        if train:
            df = df_train
            print(f"Training set: {len(df)} subjects")
        else:
            df = df_val
            print(f"Validation set: {len(df)} subjects")

        # Store the split dataframe so it can be saved externally
        self.split_df = df

        # Build subjects list
        subjects = []
        for _, row in df.iterrows():
            mask_path = os.path.join(self.data_dir, row['mask_file'])
            context = np.array([row[k] if pd.notna(row[k]) else 0.0
                                for k in self.keys], dtype=np.float32)
            subject = tio.Subject(
                mask=tio.LabelMap(mask_path),
                context=torch.from_numpy(context),
            )
            subjects.append(subject)

        self.augment = augment
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(
                    scales=1,
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5,
                ),
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)