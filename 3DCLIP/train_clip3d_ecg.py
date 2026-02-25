import os

from matplotlib.pylab import spacing
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from model import CLIP
from clip_dataloader import clip3d_ecg_dataset
import torchio as tio
import wandb

from tqdm import tqdm
import time


def train():
    # Training settings
    save_every = 20
    wandb_bool = True
    epochs = 1000
    batch_size = 32
    learning_rate = 1e-4

    # 1.0mm spacing, 192³ network, batch 32 is the clear winner:
    # 100% of scans (all ~5065 fit)
    # Batch 32 (great for contrastive learning)
    # 56 GB VRAM (plenty of headroom)
    # 1.0mm is standard resolution for cardiac CT — no meaningful information loss for fat masks

    # Set the device
    if torch.cuda.is_available():
        # Select the second GPU (index 1)
        device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    else:
        device = torch.device("cpu")

    # --- Model architecture ---
    embed_dim = 128
    image_resolution = 192         # 192³ input (must be divisible by 32)
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

    if wandb_bool:
        wandb.init(
            project="CLIP3D-ECG",
            entity="andreasaspe",
            notes="Back to old commit - major-violet-9. Training on full data to see how it works.",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embed_dim": embed_dim,
                "image_resolution": image_resolution,
                "vision_layers": vision_layers,
                "vision_width": vision_width,
                "context_length": context_length,
                "transformer_width": transformer_width,
                "transformer_heads": transformer_heads,
                "transformer_layers": transformer_layers,
            }
        )
        wandb.watch(model)


    # --- Paths ---
    out_dir = "/data/awias/NLDL_Winterschool/models/"
    data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
    csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan.csv"
    os.makedirs(out_dir, exist_ok=True)

    if wandb_bool:
        out_dir = os.path.join(out_dir, wandb.run.name)
        os.makedirs(out_dir, exist_ok=True)
    else:
        # Generate a unique folder name based on timestamp
        timestamp = time.strftime("%d.%m.%Y-%H:%M:%S")
        out_dir = os.path.join(out_dir, f"{timestamp}")
        os.makedirs(out_dir, exist_ok=True)

    # --- Optional checkpoint resume ---
    model_chkpt = None
    optimizer_chkpt = None

    # --- Datasets (80/20 split from CSV, same seed) ---
    ds_tr = clip3d_ecg_dataset(data_dir, csv_path, augment=True,  train=True)
    ds_val = clip3d_ecg_dataset(data_dir, csv_path, augment=False, train=False)

    dl_tr = tio.SubjectsLoader(ds_tr,
                               batch_size=batch_size,
                               num_workers=0,
                               shuffle=True)
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=False)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if model_chkpt is not None:
        model.load_state_dict(model_chkpt)
    if optimizer_chkpt is not None:
        optimizer.load_state_dict(optimizer_chkpt)

    scaler = GradScaler()


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
                wandb.log({
                    "train/loss": loss.item(),
                    "train/moving_avg_loss": total_loss / total_steps,
                    "epoch": epoch
                })

        avg_loss = epoch_loss / len(dl_tr)
        if wandb_bool:
            wandb.log({
                "train/avg_epoch_loss": avg_loss,
                "epoch": epoch
            })

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
            wandb.log({
                "validation/avg_epoch_loss": avg_val_loss,
                "epoch": epoch
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       os.path.join(out_dir, 'best_clip3d_ecg.pth'))


if __name__ == "__main__":
    train()
