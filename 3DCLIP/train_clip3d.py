import os 

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from model import CLIP  # Assuming you have a CLIP model defined in model.py
from clip_dataloader import clip3d_dataloader, clip3d_subjects_dataset  # Assuming you have a dataloader defined in clip_dataloader.py
import torchio as tio
import wandb

from tqdm import tqdm

def train():
    # Training settings
    save_every = 20  # Save model every 10 epochs
    wandb_bool = True  # Set to True if you want to use Weights & Biases for logging
    epochs = 200
    batch_size = 32
    learning_rate = 1e-4


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    out_dir = "/data/Data/bjorn/models/clip3d/quantile_0.01_0.99/"
    data_dir = "/data/Data/cropped_laa128_64mm/masks"
    json_dir_tr = "/data/Data/laa_measures/train_0.01_0.99/quantile_normalized.json"
    json_dir_val = "/data/Data/laa_measures/test_0.01_0.99/quantile_normalized.json"
    os.makedirs(out_dir, exist_ok=True)

    model_chkpt = None #torch.load("/data/Data/bjorn/models/clip3d/quantile_128/clip3d_epoch_29.pth")
    optimizer_chkpt = None #torch.load("/data/Data/bjorn/models/clip3d/quantile_128/optimizer-ep-29.pth")

    ds_tr = clip3d_subjects_dataset(data_dir, json_dir_tr, augment=True)
    ds_val = clip3d_subjects_dataset(data_dir, json_dir_val)
    dl_tr = tio.SubjectsLoader(ds_tr,
                           batch_size=batch_size,
                           num_workers=32,
                           shuffle=True)
    dl_val = tio.SubjectsLoader(ds_val,
                            batch_size=batch_size,
                            num_workers=32,
                            shuffle=False)

    # Initialize optimizer, scaler for FP16
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    if model_chkpt is not None:
        model.load_state_dict(model_chkpt)
    if optimizer_chkpt is not None:
        optimizer.load_state_dict(optimizer_chkpt)
    
    scaler = GradScaler()

    # Assuming you already have a dataloader
    # dataloader yields: images [batch_size, 1, 128, 128, 128], context_vectors [batch_size, 13]

    if wandb_bool:
        wandb.init(project="CLIP3D", entity="Bjonze")

    total_loss = 0.0
    total_steps = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):

        model.train()  # set model to training mode
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dl_tr), total=len(dl_tr))
        progress_bar.set_description(f'Epoch: {epoch+1}')
        
        for step, batch in progress_bar:
            images, context_vectors = batch['mask'][tio.DATA], batch['context']
            images = images.to(device, non_blocking=True)
            context_vectors = context_vectors.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():  # FP16 mixed precision
                logits_per_image, logits_per_context = model(images, context_vectors)
                
                # Ground truth labels for contrastive loss
                labels = torch.arange(images.size(0), device=device)
                
                # Compute cross-entropy loss for symmetric contrastive learning
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_ctx = F.cross_entropy(logits_per_context, labels)
                
                loss = (loss_img + loss_ctx) / 2.0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            total_loss += loss.item()
            total_steps += 1

            if wandb_bool:
                wandb.log({"train/loss": loss.item(), "train/moving_avg_loss": total_loss / total_steps})

        avg_loss = epoch_loss / len(dl_tr)
        if wandb_bool:
            wandb.log({"train/avg_epoch_loss": avg_loss})

        # Optionally save checkpoints periodically
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f'clip3d_epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(out_dir, f'optimizer-ep-{epoch}.pth'))

        # Validation step
        model.eval()
        val_loss = 0.0
        progress_bar_val = tqdm(enumerate(dl_val), total=len(dl_val))
        progress_bar_val.set_description(f'Validation after epoch: {epoch+1}')

        with torch.no_grad():
            for step, batch in progress_bar_val:
                images, context_vectors = batch['mask'][tio.DATA], batch['context']
                images = images.to(device, non_blocking=True)
                context_vectors = context_vectors.to(device, non_blocking=True)

                with autocast():
                    logits_per_image, logits_per_context = model(images, context_vectors)
                    labels = torch.arange(images.size(0), device=device)
                    loss_img = F.cross_entropy(logits_per_image, labels)
                    loss_ctx = F.cross_entropy(logits_per_context, labels)
                    loss = (loss_img + loss_ctx) / 2.0

                val_loss += loss.item()
        avg_val_loss = val_loss / len(dl_val)
        if wandb_bool:
            wandb.log({"validation/avg_epoch_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_clip3d.pth'))


if __name__ == "__main__":
    train()