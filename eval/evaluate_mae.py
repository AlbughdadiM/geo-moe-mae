"""
A class to evalute the reconstuction of MoE-MAE.
It reconstruct the unmasked and masked version of
an image and returns the different loss values.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_config import BigEarthNetInfo
from utils.data_utils import (
    load_model,
    show_side_by_side,
    unnormalize,
)
from datasets.bigearthnet import BigEarthNetDatasetLS
from models.moe_mae import mmLiT, build_model


class MAEEvaluator:
    def __init__(self, config, transform):
        self.config = config
        self.device = torch.device(config["model"]["device"])
        encoder = build_model(
            size=config["model"]["size"],
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_channels"],
        )
        model = mmLiT(encoder)
        self.model = load_model(model, config["model"]["checkpoint_path"], self.device)
        self.dataset = BigEarthNetDatasetLS(
            config["data"]["dataset_csv"],
            config["data"]["dataset_path"],
            transform=transform,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=False,
        )

    @torch.no_grad()
    def reconstruct_image(
        self,
        patches: torch.Tensor,
        patch_size: int,
        img_size: int,
    ):
        """
        Reconstructs an image from a tensor of patches using the restore indices.
        This is the key to creating a visually correct image from the model's output.

        Args:
            patches: Tensor of shape [B, num_patches, patch_dim]
            ids_restore: Tensor of shape [B, num_patches] mapping each predicted patch to its original position
            patch_size: Size of each square patch (int)
            img_size: Size of the original square image (int)

        Returns:
            Reconstructed image tensor of shape [B, C, img_size, img_size]
        """
        B = patches.shape[0]

        # Fold patches back into full image
        img = F.fold(
            patches.transpose(1, 2),  # [B, patch_dim, num_patches]
            output_size=(img_size, img_size),
            kernel_size=patch_size,
            stride=patch_size,
        )

        return img

    @torch.no_grad()
    def _evaluate_model(
        self, save_recon_masked=False, save_recon_full=False, recon_dir=None
    ):
        self.model.eval()
        masked_losses = []
        unmasked_losses = []
        moe_losses = []
        total_losses = []
        full_losses = []

        alpha = 0.1
        beta = 0.5

        pbar = tqdm(self.loader, desc="Evaluating")
        idx = 0
        for imgs, _, meta_week, meta_hour, meta_lat, meta_lon in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            meta_week = meta_week.to(self.device, non_blocking=True)
            meta_hour = meta_hour.to(self.device, non_blocking=True)
            meta_lat = meta_lat.to(self.device, non_blocking=True)
            meta_lon = meta_lon.to(self.device, non_blocking=True)

            # --- Masked evaluation ---
            self.model.mask_ratio = 0.75
            pred, mask, ids_restore, moe_loss = self.model(
                imgs,
                meta_week=meta_week,
                meta_hour=meta_hour,
                meta_lat=meta_lat,
                meta_lon=meta_lon,
            )

            # Unfold target into patches
            target_patches = F.unfold(
                imgs,
                kernel_size=self.model.encoder.patch_size,
                stride=self.model.encoder.patch_size,
            ).transpose(1, 2)

            num_masked = mask.sum()
            num_unmasked = (1 - mask).sum()

            if num_masked > 0:
                loss_masked = (
                    (pred - target_patches) ** 2 * mask.unsqueeze(-1)
                ).sum() / num_masked
            else:
                loss_masked = torch.zeros(1, device=self.device)

            if num_unmasked > 0:
                loss_unmasked = (
                    (pred - target_patches) ** 2 * (1 - mask).unsqueeze(-1)
                ).sum() / num_unmasked
            else:
                loss_unmasked = torch.zeros(1, device=self.device)

            loss_total = loss_masked + alpha * loss_unmasked + beta * moe_loss

            masked_losses.append(loss_masked.item())
            unmasked_losses.append(loss_unmasked.item())
            moe_losses.append(moe_loss.item())
            total_losses.append(loss_total.item())

            if save_recon_masked:
                pred_imgs = self.reconstruct_image(
                    pred, self.model.encoder.patch_size, imgs.size(-1)
                )
                pred_img_unnorm = unnormalize(
                    pred_imgs,
                    BigEarthNetInfo.STATISTICS["mean"],
                    BigEarthNetInfo.STATISTICS["std"],
                )
                target_img_unnorm = unnormalize(
                    imgs,
                    BigEarthNetInfo.STATISTICS["mean"],
                    BigEarthNetInfo.STATISTICS["std"],
                )
                for i in range(imgs.size(0)):
                    show_side_by_side(
                        target_img_unnorm[i],
                        pred_img_unnorm[i],
                        os.path.join(recon_dir, f"result_masked_{i}_{idx}.png"),
                        title1="Original",
                        title2="Reconstruction",
                    )

            # --- Full reconstruction ---
            self.model.mask_ratio = 0.0
            pred_full, _, ids_restore, _ = self.model(
                imgs,
                meta_week=meta_week,
                meta_hour=meta_hour,
                meta_lat=meta_lat,
                meta_lon=meta_lon,
            )

            target_patches_full = F.unfold(
                imgs,
                kernel_size=self.model.encoder.patch_size,
                stride=self.model.encoder.patch_size,
            ).transpose(1, 2)

            loss_full = ((pred_full - target_patches_full) ** 2).mean()
            full_losses.append(loss_full.item())

            if save_recon_full:
                pred_imgs = self.reconstruct_image(
                    pred_full, self.model.encoder.patch_size, imgs.size(-1)
                )
                pred_img_unnorm = unnormalize(
                    pred_imgs,
                    BigEarthNetInfo.STATISTICS["mean"],
                    BigEarthNetInfo.STATISTICS["std"],
                )
                target_img_unnorm = unnormalize(
                    imgs,
                    BigEarthNetInfo.STATISTICS["mean"],
                    BigEarthNetInfo.STATISTICS["std"],
                )
                for i in range(imgs.size(0)):
                    show_side_by_side(
                        target_img_unnorm[i],
                        pred_img_unnorm[i],
                        os.path.join(recon_dir, f"result_full_{i}_{idx}.png"),
                        title1="Original",
                        title2="Reconstruction",
                    )
            idx += 1

        return {
            "masked_loss": sum(masked_losses) / len(masked_losses),
            "unmasked_loss": sum(unmasked_losses) / len(unmasked_losses),
            "moe_loss": sum(moe_losses) / len(moe_losses),
            "total_loss": sum(total_losses) / len(total_losses),
            "full_loss": sum(full_losses) / len(full_losses),
        }

    def run(self):
        recon_dir = self.config["eval"]["reconstructions_dir"]
        os.makedirs(recon_dir, exist_ok=True)
        save_recon_masked = self.config["eval"]["save_reconstructions_masked"]
        save_recon_full = self.config["eval"]["save_reconstructions_full"]
        metrics = self._evaluate_model(
            save_recon_masked=save_recon_masked,
            save_recon_full=save_recon_full,
            recon_dir=recon_dir,
        )
        return metrics
