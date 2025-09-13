"""
A script to pretrain MoE-MAE.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import argparse
import csv
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from tqdm import tqdm

from datasets.bigearthnet import BigEarthNetDatasetLS
from models.moe_mae import build_model, MOEMAE
from scheduler.schedulers import WarmupCosineLR
from transformation.transformer import CenterCrop40, ToFloat, ZScoreNormalize
from utils.data_config import BigEarthNetInfo

# Hyperparameters for loss components
ALPHA = 0.1
BETA = 0.5


def train_epoch(
    model, loader, opt, scaler, alpha, beta, device, epoch, scheduler, total_epochs
):
    """
    Performs one epoch of training, including gradient clipping and mixed precision.
    """
    model.train()

    total_loss = 0.0
    total_loss_masked = 0.0
    total_loss_unmasked = 0.0
    total_moe_loss = 0.0

    step_in_epoch = 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch + 1}/{total_epochs}")
    lr_log = []

    for imgs, _, meta_week, meta_hour, meta_lat, meta_lon in pbar:
        global_step = epoch * len(loader) + step_in_epoch
        lr = scheduler.step(global_step)
        lr_log.append(lr)

        imgs = imgs.to(device, non_blocking=True)
        meta_week = meta_week.to(device, non_blocking=True)
        meta_hour = meta_hour.to(device, non_blocking=True)
        meta_lat = meta_lat.to(device, non_blocking=True)
        meta_lon = meta_lon.to(device, non_blocking=True)

        opt.zero_grad()
        use_amp = device == "cuda"
        with torch.amp.autocast(enabled=use_amp, device_type=device):
            pred, mask, _, moe_loss = model(
                imgs,
                meta_week=meta_week,
                meta_hour=meta_hour,
                meta_lat=meta_lat,
                meta_lon=meta_lon,
            )
            target_patches = F.unfold(
                imgs,
                kernel_size=model.encoder.patch_size,
                stride=model.encoder.patch_size,
            ).transpose(1, 2)
            num_masked = mask.sum()
            num_unmasked = (1 - mask).sum()

            if num_masked > 0:
                loss_masked = (
                    (pred - target_patches) ** 2 * mask.unsqueeze(-1)
                ).sum() / num_masked
            else:
                loss_masked = torch.zeros(1, device=device)

            if num_unmasked > 0:
                loss_unmasked = (
                    (pred - target_patches) ** 2 * (1 - mask).unsqueeze(-1)
                ).sum() / num_unmasked
            else:
                loss_unmasked = torch.zeros(1, device=device)

            # Weighted sum of losses
            loss = loss_masked + alpha * loss_unmasked + beta * moe_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected at step {global_step}. Stopping training.")
            return None, None, None, None, None

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        total_loss_masked += loss_masked.item()
        total_loss_unmasked += loss_unmasked.item()
        total_moe_loss += moe_loss.item()
        step_in_epoch += 1
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "masked": f"{loss_masked.item():.4f}",
                "unmasked": f"{loss_unmasked.item():.4f}",
                "moe": f"{moe_loss.item():.4f}",
                "lr": f"{lr:.6f}",
            }
        )

    return (
        total_loss / len(loader),
        total_loss_masked / len(loader),
        total_loss_unmasked / len(loader),
        total_moe_loss / len(loader),
        sum(lr_log) / len(lr_log),
    )


def main():
    """
    The main training loop.
    """
    parser = argparse.ArgumentParser(description="Training MAE MMLiT on BigEarthNet")
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint if available"
    )
    args = parser.parse_args()

    with open(args.config_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    scaler = torch.amp.GradScaler(device=device)

    encoder = build_model(
        size=config["model"]["size"],
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=config["model"]["in_channels"],
    )
    model = MOEMAE(encoder).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    bigearth_transforms = transforms.Compose(
        [
            ToFloat(),
            CenterCrop40(),
            ZScoreNormalize(
                BigEarthNetInfo.STATISTICS["mean"],
                BigEarthNetInfo.STATISTICS["std"],
            ),
        ]
    )
    train_dataset = BigEarthNetDatasetLS(
        config["training"]["dataset_csv"],
        config["training"]["dataset_path"],
        transform=bigearth_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        persistent_workers=False,
        prefetch_factor=4,
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        pin_memory=True,
    )
    save_path = config["training"]["weight_path"]
    os.makedirs(save_path, exist_ok=True)

    start_epoch = 0
    warmup_epochs = int(0.05 * config["training"]["epochs"])
    best_train_loss = float("inf")

    ckpt_path = f"{save_path}/checkpoint_{config['model']['size']}.pth"
    if args.resume and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["opt_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        print(
            f"Resumed from epoch {start_epoch}, best training loss {best_train_loss:.4f}"
        )
    total_steps = int(config["training"]["epochs"] * len(train_loader))
    warmup_steps = int(warmup_epochs * len(train_loader))
    scheduler = WarmupCosineLR(
        opt, total_steps, config["training"]["learning_rate"], warmup_steps
    )
    save_every_n_epochs = config["training"].get("save_every_n_epochs", 5)
    loss_log_path = f"{save_path}/training_metrics_{config['model']['size']}.csv"
    log_mode = "a" if start_epoch > 0 else "w"
    with open(loss_log_path, log_mode, newline="", encoding="utf-8") as loss_log_file:
        csv_writer = csv.writer(loss_log_file)
        if start_epoch == 0:
            csv_writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_m_loss",
                    "train_um_loss",
                    "train_moe_loss",
                    "avg_lr",
                ]
            )

        for ep in range(start_epoch, config["training"]["epochs"]):
            (
                train_loss,
                train_masked_loss,
                train_unmasked_loss,
                train_moe_loss,
                avg_lr,
            ) = train_epoch(
                model,
                train_loader,
                opt,
                scaler,
                ALPHA,
                BETA,
                device,
                ep,
                scheduler,
                config["training"]["epochs"],
            )

            if train_loss is None:
                break

            csv_writer.writerow(
                [
                    ep,
                    f"{train_loss:.4f}",
                    f"{train_masked_loss:.4f}",
                    f"{train_unmasked_loss:.4f}",
                    f"{train_moe_loss:.4f}",
                    f"{avg_lr:.8f}",
                ]
            )
            print(
                f"Epoch {ep+1}/{config['training']['epochs']} | Train Loss {train_loss:.4f} | Avg LR {avg_lr:.8f}"
            )

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(
                    {
                        "epoch": ep,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_train_loss": best_train_loss,
                    },
                    f"{save_path}/pretrained_{config['model']['size']}_best.pth",
                )
                print(f"New best model saved with training loss: {best_train_loss:.4f}")

            if (ep + 1) % save_every_n_epochs == 0:
                torch.save(
                    {
                        "epoch": ep,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_train_loss": best_train_loss,
                    },
                    ckpt_path,
                )
            loss_log_file.flush()
    torch.save(
        {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_train_loss": best_train_loss,
        },
        f"{save_path}/pretrained_{config['model']['size']}_last.pth",
    )
    print("Training complete. Last epoch model saved.")


if __name__ == "__main__":
    main()
