import torch 
import numpy as np
import os

import torch.utils 
import json
import SimpleITK as sitk
import torchio as tio

class clip3d_dataloader(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_dir, augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.augment = augment
        self.json_file = load_results_from_json(json_dir)
        self.data_list = os.listdir(data_dir)
        #self.keys = ["laa_lobes_n", "laa_volume", "laa_orifice_area",
        #            "laa_orifice_long", "laa_orifice_short", "sph_idx", "heigth", "weigth", "age"]
        self.keys = ["tortuosity", "centerline_length", "max_geodesic_distance", "volume",
                     "angle_ostium_laa", "cl_cut_25_elongation", "cl_cut_25_cutarea", "cl_cut_50_elongation", "cl_cut_50_cutarea",
                     "cl_cut_75_elongation", "cl_cut_75_cutarea", "radii_95", "normalized_shape_index",
                     "elongation", "flatness", "surface_area", "ostium_major_axis_length", "ostium_minor_axis_length"]
        if self.augment:
            # Compose rigid and flip transforms
            self.aug_transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5), # Random flip on any axis
                tio.RandomAffine(
                    scales=1,  # No scaling
                    degrees=30,  # Up to 45 degrees random rotation
                    translation=10,  # Up to 10 voxels random translation
                    isotropic=True,  # Isotropic scaling
                    center='image',  # Random center for rotation
                    default_pad_value=0,  # Padding value
                    image_interpolation='nearest',
                    p=0.5
                ),
            ])
        else:
            self.aug_transform = None
        
        
    def __len__(self) -> int:
        self.len = len(self.json_file)
        return self.len
    
    def __getitem__(self, idx):
        json_dict = self.json_file[idx]
        file_name = json_dict["filename"]
        file_name = file_name.replace(".nii.gz", "_labels.nii.gz")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_dir, file_name)))
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        if self.aug_transform:
            mask = self.aug_transform(mask)

        context = [json_dict[key] for key in self.keys]
        context = [-1 if (c == "None" or np.isnan(c)) else c for c in context]
        context = np.array(context, dtype=np.float32)
        context = torch.from_numpy(context)

        return mask, context


class clip3d_subjects_dataset(tio.SubjectsDataset):
    def __init__(self, data_dir, json_dir, augment=False):
        self.data_dir = data_dir
        self.json_file = load_results_from_json(json_dir)
        self.keys = ["tortuosity", "centerline_length", "max_geodesic_distance", "volume",
                     "angle_ostium_laa", "cl_cut_25_elongation", "cl_cut_25_cutarea", "cl_cut_50_elongation", "cl_cut_50_cutarea",
                     "cl_cut_75_elongation", "cl_cut_75_cutarea", "radii_95", "normalized_shape_index",
                     "elongation", "flatness", "surface_area", "ostium_major_axis_length", "ostium_minor_axis_length"]
        subjects = []
        for json_dict in self.json_file:
            file_name = json_dict["filename"].replace(".nii.gz", "_labels.nii.gz")
            mask_path = os.path.join(self.data_dir, file_name)
            context = [json_dict[key] for key in self.keys]
            context = [-1 if (c == "None" or np.isnan(c)) else c for c in context]
            context = np.array(context, dtype=np.float32)

            # Create a Subject
            subject = tio.Subject(
                mask=tio.LabelMap(mask_path),
                context=torch.from_numpy(context)
            )
            subjects.append(subject)

        self.augment = augment
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(
                    scales=1,  # No scaling
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5
                ),
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)  # This returns a Subject!


class clip3d_subjects_withName(tio.SubjectsDataset):
    def __init__(self, data_dir, json_dir, augment=False):
        self.data_dir = data_dir
        self.json_file = load_results_from_json(json_dir)
        self.keys = ["tortuosity", "centerline_length", "max_geodesic_distance", "volume",
                     "angle_ostium_laa", "cl_cut_25_elongation", "cl_cut_25_cutarea", "cl_cut_50_elongation", "cl_cut_50_cutarea",
                     "cl_cut_75_elongation", "cl_cut_75_cutarea", "radii_95", "normalized_shape_index",
                     "elongation", "flatness", "surface_area", "ostium_major_axis_length", "ostium_minor_axis_length"]
        subjects = []
        for json_dict in self.json_file:
            file_name = json_dict["filename"].replace(".nii.gz", "_labels.nii.gz")
            mask_path = os.path.join(self.data_dir, file_name)
            context = [json_dict[key] for key in self.keys]
            context = [-1 if (c == "None" or np.isnan(c)) else c for c in context]
            context = np.array(context, dtype=np.float32)

            # Create a Subject
            subject = tio.Subject(
                mask=tio.LabelMap(mask_path),
                context=torch.from_numpy(context),
                name = file_name
            )
            subjects.append(subject)

        self.augment = augment
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(
                    scales=1,  # No scaling
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5
                ),
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)  # This returns a Subject!


class clip3d_score_dataset(tio.SubjectsDataset):
    def __init__(self, data_dir, json_dir, augment=False):
        self.data_dir = data_dir
        self.json_file = load_results_from_json(json_dir)
        self.keys = ["tortuosity", "centerline_length", "max_geodesic_distance", "volume",
                     "angle_ostium_laa", "cl_cut_25_elongation", "cl_cut_25_cutarea", "cl_cut_50_elongation", "cl_cut_50_cutarea",
                     "cl_cut_75_elongation", "cl_cut_75_cutarea", "radii_95", "normalized_shape_index",
                     "elongation", "flatness", "surface_area", "ostium_major_axis_length", "ostium_minor_axis_length"]
        subjects = []
        for json_dict in self.json_file:
            file_name = json_dict["filename"].replace(".nii.gz", "_labels.nii.gz")
            mask_path = os.path.join(self.data_dir, file_name)
            context = [json_dict[key] for key in self.keys]
            context = [-1 if (c == "None" or np.isnan(c)) else c for c in context]
            context = np.array(context, dtype=np.float32)
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file {mask_path} does not exist. Skipping this entry.")
                continue

            # Create a Subject
            subject = tio.Subject(
                mask=tio.LabelMap(mask_path),
                context=torch.from_numpy(context)
            )
            subjects.append(subject)

        self.augment = augment
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(
                    scales=1,  # No scaling
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5
                ),
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)  # This returns a Subject!


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
        if train:
            df = df.iloc[:split_idx].reset_index(drop=True)
            print(f"Training set: {len(df)} subjects")
        else:
            df = df.iloc[split_idx:].reset_index(drop=True)
            print(f"Validation set: {len(df)} subjects")

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
                    scales=(0.95, 1.05),     # mild random scaling
                    degrees=15,              # up to 15° rotation (was 10)
                    translation=5,
                    isotropic=False,
                    image_interpolation='nearest',
                    p=0.5,
                ),
                tio.RandomElasticDeformation(  # small elastic warps
                    num_control_points=5,
                    max_displacement=4,
                    image_interpolation='nearest',
                    p=0.3,
                ),
                tio.RandomNoise(std=(0, 0.05), p=0.2),  # light Gaussian noise
            ])
        else:
            self.transform = None

        super().__init__(subjects, transform=self.transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def load_results_from_json(input_path):
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"Data successfully loaded from {input_path}")
        return data
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Please check the file path and try again.")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}. Please ensure the file contains valid JSON.")