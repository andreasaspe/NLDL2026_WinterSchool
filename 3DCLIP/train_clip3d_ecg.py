import os
import math

import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
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
    learning_rate = 5e-5             # lower peak LR — scheduler will warm up to it
    weight_decay = 1e-2              # strong AdamW regularisation (was 1e-5)
    warmup_epochs = 10               # linear LR warmup before cosine decay
    label_smoothing = 0.1            # prevents overconfident logits
    early_stop_patience = 60         # stop if val loss stalls for this many epochs

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
            notes="Working commit (aka major-violet-9), but now claude tried to fix overfit - and on full data. Let's see what happens.",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": label_smoothing,
                "early_stop_patience": early_stop_patience,
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
                               shuffle=True,
                               pin_memory=True,
                               drop_last=True)      # critical for contrastive loss
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=batch_size,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-6)
    if model_chkpt is not None:
        model.load_state_dict(model_chkpt)
    if optimizer_chkpt is not None:
        optimizer.load_state_dict(optimizer_chkpt)

    # --- LR Scheduler: linear warmup + cosine decay ---
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs                  # linear 0→1
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * progress))  # decay to 1% of peak
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Lower init_scale — default 2^16 overflows with 192³ FP16 convolutions
    scaler = GradScaler(init_scale=2**10, growth_interval=1000)


    total_loss = 0.0
    total_steps = 0
    best_val_loss = float('inf')
    nan_count = 0              # track consecutive NaN batches
    epochs_no_improve = 0      # early stopping counter

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        progress_bar = tqdm(enumerate(dl_tr), total=len(dl_tr))
        progress_bar.set_description(f'Epoch: {epoch+1}')

        for step, batch in progress_bar:
            images = batch['mask'][tio.DATA].to(device, non_blocking=True)
            context_vectors = batch['context'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits_per_image, logits_per_context = model(images, context_vectors)

                # Check for NaN in logits (FP16 overflow)
                if torch.isnan(logits_per_image).any() or torch.isnan(logits_per_context).any():
                    nan_count += 1
                    print(f"  ⚠ NaN in logits at epoch {epoch+1}, step {step} "
                          f"(logit_scale={model.logit_scale.item():.3f}, "
                          f"exp={model.logit_scale.clamp(max=4.6052).exp().item():.1f}) — skipping")
                    continue

                labels = torch.arange(images.size(0), device=device)
                loss_img = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
                loss_ctx = F.cross_entropy(logits_per_context, labels, label_smoothing=label_smoothing)
                loss = (loss_img + loss_ctx) / 2.0

            # Skip if loss is NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  ⚠ NaN/Inf loss at epoch {epoch+1}, step {step} — skipping")
                continue

            nan_count = 0  # reset on clean step

            scaler.scale(loss).backward()

            # Unscale before clipping so we clip the true gradient magnitude
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Skip step if gradients are still broken after clipping
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"  ⚠ Inf/NaN grads at epoch {epoch+1}, step {step} — skipping step")
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            total_loss += loss.item()
            total_steps += 1
            epoch_steps += 1

            if wandb_bool:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/moving_avg_loss": total_loss / total_steps,
                    "train/logit_scale": model.logit_scale.item(),
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })

        # Step LR scheduler after each epoch
        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1}: train_loss={avg_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

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
        val_steps = 0
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

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        if wandb_bool:
            wandb.log({
                "validation/avg_epoch_loss": avg_val_loss,
                "epoch": epoch
            })

        # --- Best-model tracking + early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(out_dir, 'best_clip3d_ecg.pth'))
            print(f"  ✓ New best model (val_loss={avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\n  ✗ Early stopping — no improvement for {early_stop_patience} epochs.")
                break

    if wandb_bool:
        wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
