
from clip_dataloader import clip3d_subjects_withName
from model import CLIP
from tqdm import tqdm
import torchio as tio
import torch
import numpy as np

@torch.no_grad()
def save_clip_latents(model, dataloader, device, save_path):
    model.eval()
    model.to(device)
    for batch in tqdm(dataloader):
        images, context_vectors, name = batch['mask'][tio.DATA], batch['context'], batch['name']
        images = images.to(device, non_blocking=True)
        context = context_vectors.to(device, non_blocking=True)

        image_features, unpooled_context_features = model.extract_features(images, context)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        unpooled_context_features = unpooled_context_features / unpooled_context_features.norm(dim=2, keepdim=True) # we normalize the context features per token

        image_features = image_features.squeeze().cpu().numpy()  # Convert to numpy array
        unpooled_context_features = unpooled_context_features.squeeze().cpu().numpy()  # Convert to numpy array

        # we save them as a npz file 
        name_withDir = os.path.join(save_path, name[0].split(".")[0]+".npz")
        np.savez(name_withDir, clip_image_features=image_features, clip_context_features=unpooled_context_features, context=context.cpu().numpy())


if __name__ == "__main__":
    import os
    save_path = "/data/Data/latent_vectors/clip3d/quantile_128_all"
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_dir_tr = "/data/Data/laa_measures/train/quantile_normalized.json"
    json_dir_val = "/data/Data/laa_measures/val/quantile_normalized.json"
    data_dir = "/data/Data/cropped_laa128_64mm/masks"


    # define data loader
    ds_tr = clip3d_subjects_withName(data_dir, json_dir_tr, augment=False)
    dl_tr = tio.SubjectsLoader(ds_tr,
                           batch_size=1,
                           num_workers=2,
                           shuffle=False)
    
    ds_val = clip3d_subjects_withName(data_dir, json_dir_val)
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

    #define model 
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
    ).to(device)
    # Load model weights
    model_path = "/data/Data/bjorn/models/clip3d/quantile_128_all/best_clip3d.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    train_save_path = os.path.join(save_path, "train")
    os.makedirs(train_save_path, exist_ok=True)
    validation_save_path = os.path.join(save_path, "validation")
    os.makedirs(validation_save_path, exist_ok=True)

    save_clip_latents(model, dl_tr, device, train_save_path)
    save_clip_latents(model, dl_val, device, validation_save_path)