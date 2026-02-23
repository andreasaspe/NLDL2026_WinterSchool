import os

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from model import CLIP
from clip_dataloader import clip3d_ecg_dataset
import torchio as tio
import wandb

from tqdm import tqdm


def train():
    # Training settings
    save_every = 20
    wandb_bool = False
    epochs = 200
    batch_size = 32
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model architecture ---
    embed_dim = 128
    image_resolution = 320         # 256³ input (must be divisible by 32)
    vision_layers = (3, 4, 6, 3)
    vision_width = 64

    context_length = 36            # 36 ECG features
    transformer_width = 256
    transformer_heads = 4
    transformer_layers = 4

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width,
        context_length, transformer_width, transformer_heads, transformer_layers,
    ).to(device)

    # --- Paths ---
    out_dir = "/data/awias/NLDL_Winterschool/models/clip3d_ecg/"
    data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
    csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"
    os.makedirs(out_dir, exist_ok=True)

    # --- Optional checkpoint resume ---
    model_chkpt = None
    optimizer_chkpt = None

    # --- Datasets (80/20 split from CSV, same seed) ---
    ds_tr = clip3d_ecg_dataset(data_dir, csv_path, augment=True,  train=True)
    ds_val = clip3d_ecg_dataset(data_dir, csv_path, augment=False, train=False)

    dl_tr = tio.SubjectsLoader(ds_tr,
                               batch_size=batch_size,
                               num_workers=32,
                               shuffle=True)
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=batch_size,
                                num_workers=32,
                                shuffle=False)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if model_chkpt is not None:
        model.load_state_dict(model_chkpt)
    if optimizer_chkpt is not None:
        optimizer.load_state_dict(optimizer_chkpt)

    scaler = GradScaler()

    if wandb_bool:
        wandb.init(project="CLIP3D-ECG", entity="Bjonze")

    total_loss = 0.0
    total_steps = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dl_tr), total=len(dl_tr))
        progress_bar.set_description(f'Epoch: {epoch+1}')

        for step, batch in progress_bar:
            images = batch['mask'][tio.DATA].to(device, non_blocking=True)
            context_vectors = batch['context'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                logits_per_image, logits_per_context = model(images, context_vectors)
                labels = torch.arange(images.size(0), device=device)
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
                wandb.log({"train/loss": loss.item(),
                           "train/moving_avg_loss": total_loss / total_steps})

        avg_loss = epoch_loss / len(dl_tr)
        if wandb_bool:
            wandb.log({"train/avg_epoch_loss": avg_loss})

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'clip3d_ecg_epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(),
                       os.path.join(out_dir, f'optimizer-ep-{epoch}.pth'))

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        progress_bar_val = tqdm(enumerate(dl_val), total=len(dl_val))
        progress_bar_val.set_description(f'Validation after epoch: {epoch+1}')

        with torch.no_grad():
            for step, batch in progress_bar_val:
                images = batch['mask'][tio.DATA].to(device, non_blocking=True)
                context_vectors = batch['context'].to(device, non_blocking=True)

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
            torch.save(model.state_dict(),
                       os.path.join(out_dir, 'best_clip3d_ecg.pth'))


if __name__ == "__main__":
    train()
