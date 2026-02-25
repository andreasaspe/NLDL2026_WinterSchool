import os
import copy
import math

import torch
import numpy as np
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from model import CLIP
from clip_dataloader import clip3d_ecg_dataset, clip3d_ecg_dataset_nosplit
import torchio as tio
import wandb

from tqdm import tqdm
import time


# ---------------------------------------------------------------------------
#  Exponential Moving Average (EMA) — maintains a smoothed copy of the model
#  that generalises better than the raw training weights.
# ---------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params (call before validation)."""
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params after validation."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
#  Cosine schedule with linear warmup
# ---------------------------------------------------------------------------
def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `min_lr_ratio * base_lr`.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear 0→1
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
#  Contrastive accuracy helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def contrastive_accuracy(logits_per_image, logits_per_context):
    """Top-1 accuracy for the contrastive matching (img→ctx and ctx→img)."""
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    acc_img = (logits_per_image.argmax(dim=1) == labels).float().mean()
    acc_ctx = (logits_per_context.argmax(dim=1) == labels).float().mean()
    return acc_img.item(), acc_ctx.item()


def train():
    # ===================== Training settings =====================
    save_every = 20
    wandb_bool = True
    epochs = 1000
    batch_size = 32
    learning_rate = 5e-5             # slightly higher peak — scheduler will decay it
    weight_decay = 1e-2              # stronger AdamW regularisation (OpenAI CLIP uses 0.2)
    warmup_epochs = 15               # linear LR warmup
    label_smoothing = 0.1            # prevents overconfident predictions
    ema_decay = 0.999                # EMA smoothing factor
    early_stop_patience = 80         # stop if val loss doesn't improve for this many epochs
    num_workers = 0                  # non-parallel data loading

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda:0")
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
            notes="ONE more effort from claude to fix it. Cosine LR schedule + EMA + early stopping + stronger regularisation.",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": label_smoothing,
                "ema_decay": ema_decay,
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
        # wandb.watch is expensive on large models — log gradients less often
        wandb.watch(model, log="gradients", log_freq=100)

    # --- Paths ---
    out_dir = "/data/awias/NLDL_Winterschool/models/"
    data_dir = "/data/awias/NLDL_Winterschool/EAT_mask_cropped_1mm"
    csv_path = "/data/awias/NLDL_Winterschool/CT_EKG_combined_pseudonymized_with_best_phase_scan_split.csv"
    os.makedirs(out_dir, exist_ok=True)

    if wandb_bool:
        out_dir = os.path.join(out_dir, wandb.run.name)
        os.makedirs(out_dir, exist_ok=True)
    else:
        timestamp = time.strftime("%d.%m.%Y-%H:%M:%S")
        out_dir = os.path.join(out_dir, f"{timestamp}")
        os.makedirs(out_dir, exist_ok=True)

    # --- Optional checkpoint resume ---
    model_chkpt = None
    optimizer_chkpt = None

    # --- Datasets (using split column from CSV) ---
    ds_tr = clip3d_ecg_dataset(data_dir, csv_path, augment=True,  split='train')
    ds_val = clip3d_ecg_dataset(data_dir, csv_path, augment=False, split='val')

    dl_tr = tio.SubjectsLoader(ds_tr,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=True,
                               pin_memory=True,
                               persistent_workers=num_workers > 0,
                               drop_last=True)  # drop incomplete — critical for contrastive loss
    dl_val = tio.SubjectsLoader(ds_val,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=num_workers > 0)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-6)
    if model_chkpt is not None:
        model.load_state_dict(model_chkpt)
    if optimizer_chkpt is not None:
        optimizer.load_state_dict(optimizer_chkpt)

    # --- LR Scheduler: linear warmup → cosine decay ---
    scheduler = cosine_warmup_schedule(optimizer, warmup_epochs, epochs, min_lr_ratio=0.01)

    # --- EMA ---
    ema = EMA(model, decay=ema_decay)

    # --- GradScaler for mixed precision ---
    # Start with a lower scale factor — default 2^16=65536 overflows immediately
    # with 192³ 3D convolutions in FP16
    scaler = GradScaler(init_scale=2**10, growth_interval=1000)

    total_loss = 0.0
    total_steps = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0          # early stopping counter

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_acc_img, epoch_acc_ctx = 0.0, 0.0
        progress_bar = tqdm(enumerate(dl_tr), total=len(dl_tr))
        progress_bar.set_description(f'Epoch: {epoch+1}')

        for step, batch in progress_bar:
            images = batch['mask'][tio.DATA].to(device, non_blocking=True)
            context_vectors = batch['context'].to(device, non_blocking=True)

            # Skip batches with NaN inputs
            if torch.isnan(context_vectors).any() or torch.isnan(images).any():
                print(f"  ⚠ NaN in inputs at epoch {epoch+1}, step {step} — skipping batch")
                continue

            optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

            with autocast():
                logits_per_image, logits_per_context = model(images, context_vectors)

                # Check for NaN in logits (FP16 overflow)
                if torch.isnan(logits_per_image).any() or torch.isnan(logits_per_context).any():
                    print(f"  ⚠ NaN in logits at epoch {epoch+1}, step {step}")
                    print(f"    logit_scale = {model.logit_scale.item():.3f} (exp={model.logit_scale.exp().item():.1f})")
                    continue

                labels = torch.arange(images.size(0), device=device)
                loss_img = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
                loss_ctx = F.cross_entropy(logits_per_context, labels, label_smoothing=label_smoothing)
                loss = (loss_img + loss_ctx) / 2.0

            # Check loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ NaN/Inf loss at epoch {epoch+1}, step {step} — skipping batch")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)

            # Gradient clipping — prevents explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Skip optimizer step if gradients exploded
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"  ⚠ Inf/NaN gradients at epoch {epoch+1}, step {step} — skipping step")
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.step(optimizer)
            scaler.update()

            # Update EMA after each optimiser step
            ema.update(model)

            # Track accuracy
            with torch.no_grad():
                ai, ac = contrastive_accuracy(logits_per_image, logits_per_context)
            epoch_acc_img += ai
            epoch_acc_ctx += ac

            epoch_loss += loss.item()
            epoch_steps += 1
            total_loss += loss.item()
            total_steps += 1

            progress_bar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

            if wandb_bool:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/moving_avg_loss": total_loss / total_steps,
                    "train/logit_scale": model.logit_scale.item(),
                    "train/logit_scale_exp": model.logit_scale.exp().item(),
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                })

        # Step the LR scheduler after each epoch
        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_acc_img = epoch_acc_img / max(epoch_steps, 1)
        avg_acc_ctx = epoch_acc_ctx / max(epoch_steps, 1)

        if wandb_bool:
            wandb.log({
                "train/avg_epoch_loss": avg_loss,
                "train/acc_img2ctx": avg_acc_img,
                "train/acc_ctx2img": avg_acc_ctx,
                "epoch": epoch,
            })

        print(f"  Epoch {epoch+1}: train_loss={avg_loss:.4f} | acc_i2c={avg_acc_img:.2%} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"logit_scale={model.logit_scale.item():.2f} (exp={model.logit_scale.exp().item():.1f})")

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'clip3d_ecg_epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(),
                       os.path.join(out_dir, f'optimizer-ep-{epoch}.pth'))

        # ---- Validation (with EMA weights) ----
        ema.apply(model)   # swap in EMA weights
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_acc_img, val_acc_ctx = 0.0, 0.0
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
                    ai, ac = contrastive_accuracy(logits_per_image, logits_per_context)
                    val_acc_img += ai
                    val_acc_ctx += ac

        ema.restore(model)  # restore training weights

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_acc_img = val_acc_img / max(val_steps, 1)
        avg_val_acc_ctx = val_acc_ctx / max(val_steps, 1)
        train_val_gap = avg_loss - avg_val_loss

        print(f"  Epoch {epoch+1}: val_loss={avg_val_loss:.4f} | val_acc_i2c={avg_val_acc_img:.2%} | "
              f"train-val gap={train_val_gap:.4f}")

        if wandb_bool:
            wandb.log({
                "validation/avg_epoch_loss": avg_val_loss,
                "validation/acc_img2ctx": avg_val_acc_img,
                "validation/acc_ctx2img": avg_val_acc_ctx,
                "train/train_val_gap": train_val_gap,
                "epoch": epoch,
            })

        # --- Best model tracking + Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save EMA weights as the best checkpoint
            ema.apply(model)
            torch.save(model.state_dict(),
                       os.path.join(out_dir, 'best_clip3d_ecg.pth'))
            ema.restore(model)
            print(f"  ✓ New best model saved (val_loss={avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\n  ✗ Early stopping triggered — no improvement for {early_stop_patience} epochs.")
                break

    if wandb_bool:
        wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()