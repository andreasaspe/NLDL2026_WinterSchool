import os 

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from model import CLIP  # Assuming you have a CLIP model defined in model.py
from clip_dataloader import clip3d_score_dataset  # Assuming you have a dataloader defined in clip_dataloader.py
import torchio as tio
import wandb

from tqdm import tqdm
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_clip3d_model(model_chkpt):
    embed_dim = 128                # Standard CLIP embedding dimension
    image_resolution = 128         # Your volume size (128³)
    vision_layers = (3, 4, 6, 3)   # Small ResNet50-like architecture (4 layers)
    vision_width = 64              # Standard width

    context_length = 18            # Number of context tokens (your shape descriptors)
    transformer_width = 128        # Width of transformer model for context encoder
    transformer_heads = 4          # Number of heads (256 dimension with 4 heads, 64 dim per head)
    transformer_layers = 4         # Moderate depth

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width,
        context_length, transformer_width, transformer_heads, transformer_layers
    ).to(DEVICE)

    if model_chkpt is not None:
        model.load_state_dict(torch.load(model_chkpt, map_location=DEVICE))
        print("Loaded model checkpoint.")
    return model

def get_validation_loader(data_dir, json_dir, batch_size):
    ds_val = clip3d_score_dataset(data_dir, json_dir)
    dl_val = tio.SubjectsLoader(ds_val,
                        batch_size=batch_size,
                        num_workers=32,
                        shuffle=False)
    
    return dl_val

def calc_clip_score(model, dl_val):
    model.eval()
    clip_scores = []
    with torch.no_grad():
        for batch in tqdm(dl_val, desc="Calculating CLIP Score"):
            images, context_vectors = batch['mask'][tio.DATA], batch['context']
            images = images.to(DEVICE)
            context_vectors = context_vectors.to(DEVICE)

            with autocast():
                cosine_clip_score = model.get_clip_score(images, context_vectors)
            clip_scores.extend(cosine_clip_score.cpu().numpy().tolist())

    return clip_scores

def visualize_clip_scores(clip_scores):
    #include mean and std in the plot title
    mean_score = sum(clip_scores) / len(clip_scores)
    std_score = (sum((x - mean_score) ** 2 for x in clip_scores) / len(clip_scores)) ** 0.5
    print(f"Mean CLIP Score: {mean_score:.4f}, Std Dev: {std_score:.4f}")
    plt.figure(figsize=(10, 6))
    plt.hist(clip_scores, bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of CLIP Scores\nMean: {mean_score:.4f}, Std Dev: {std_score:.4f}')
    plt.xlabel('CLIP Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main():
    # Paths
    data_dir = "/data/Data/generated_validation"
    json_dir_val = "/data/Data/laa_measures/test_0.01_0.99/quantile_normalized.json"
    model_chkpt = "/data/Data/bjorn/models/clip3d/quantile_0.01_0.99/best_clip3d.pth"

    # Load model
    model = get_clip3d_model(model_chkpt)

    # Get validation data loader
    batch_size = 32
    dl_val = get_validation_loader(data_dir, json_dir_val, batch_size)

    # Calculate CLIP scores
    clip_scores = calc_clip_score(model, dl_val)

    # Visualize CLIP scores
    visualize_clip_scores(clip_scores)

if __name__ == "__main__":
    main()