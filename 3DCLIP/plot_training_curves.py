import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_log_path = "/data/awias/NLDL_Winterschool/training_log.csv"
validation_log_path = "/data/awias/NLDL_Winterschool/validation_log.csv"
output_folder = "/data/awias/NLDL_Winterschool/training_curves"
os.makedirs(output_folder, exist_ok=True)

# Load data (second column from each file)
train_df = pd.read_csv(training_log_path)
val_df = pd.read_csv(validation_log_path)

train_values = train_df.iloc[:, 1].values.astype(float)
val_values = val_df.iloc[:, 1].values.astype(float)
epochs_train = np.arange(1, len(train_values) + 1)
epochs_val = np.arange(1, len(val_values) + 1)

# ---- Plot style ----
plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'figure.dpi': 150,
})

# ---- Training curve (train/val gap) ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_train, train_values, color='#2196F3', label='Train-Val Gap')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# ax.set_title('Training Curve')
# ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(output_folder, 'training_curve.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(output_folder, 'training_curve.png')}")

# ---- Validation curve (avg epoch loss) ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_val, val_values, color='#F44336', label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# ax.set_title('Validation Curve')
# ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(output_folder, 'validation_curve.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(output_folder, 'validation_curve.png')}")

plt.show()
