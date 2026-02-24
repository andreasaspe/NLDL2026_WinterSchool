import pandas as pd
import numpy as np
from pathlib import Path

def create_train_val_test_split(csv_path, output_path=None, random_seed=42):
    """
    Create 80/20 train/test split, then split train into 80/20 train/val.
    
    Final splits: ~64% train, ~16% val, ~20% test
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file (optional, defaults to input with '_split' suffix)
        random_seed: Random seed for reproducibility
    """
    # Read the CSV
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # First split: 80% train+val, 20% test
    n_total = len(df)
    n_test = int(0.2 * n_total)
    n_train_val = n_total - n_test
    
    # Second split: from train+val, split 80% train, 20% val
    n_train = int(0.8 * n_train_val)
    n_val = n_train_val - n_train
    
    # Assign splits
    split_labels = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    df['split'] = split_labels
    
    # Print statistics
    print("\n" + "="*50)
    print("Split Statistics:")
    print("="*50)
    print(f"Train: {n_train} samples ({n_train/n_total*100:.1f}%)")
    print(f"Val:   {n_val} samples ({n_val/n_total*100:.1f}%)")
    print(f"Test:  {n_test} samples ({n_test/n_total*100:.1f}%)")
    print(f"Total: {n_total} samples")
    print("="*50)
    
    # Value counts verification
    print("\nValue counts:")
    print(df['split'].value_counts().sort_index())
    
    # Determine output path
    if output_path is None:
        input_path = Path(csv_path)
        output_path = input_path.parent / f"{input_path.stem}_split{input_path.suffix}"

    # Move split column to third column after pseudo_id and nifti columns
    cols = df.columns.tolist()
    cols.insert(2, cols.pop(cols.index('split')))
    df = df[cols]
    
    # Save the enriched CSV
    print(f"\nSaving enriched CSV to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")
    
    return df


if __name__ == "__main__":
    # Input CSV path
    csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"
    
    # Create the split
    df = create_train_val_test_split(csv_path, random_seed=42)
