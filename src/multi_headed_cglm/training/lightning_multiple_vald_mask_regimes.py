import os
import sys
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from ml_collections import ConfigDict

from .orchestrator import SingleStemCriticPretrainingOrchestrator


class LightningSingleStemCriticPretrain(pl.LightningModule):
    """
    Lightning module for single-stem multi-head critic pretraining with multiple validation mask configurations.

    - Uses the batch preparer to build (dna, exp_tracks, pred_tracks) batches.
    - Uses SingleStemCriticPretrainingOrchestrator to compute total loss and per-head losses.
    - Supports multiple validation batch preparers (dict) for evaluating on different mask rate bins.
    - Each validation step evaluates the model on all mask configurations and logs separately.
    - Single stem (no regime routing), multiple heads (GT, S2F, DISC).
    """

    def __init__(
        self,
        config_dict: ConfigDict,
        critic: torch.nn.Module,
        orchestrator: SingleStemCriticPretrainingOrchestrator,
        train_batch_preparer,
        val_batch_preparers: Dict[str, Any],
    ):
        super().__init__()
        self.config_dict = config_dict
        self.critic = critic
        self.orchestrator = orchestrator
        self.train_batch_preparer = train_batch_preparer
        self.val_batch_preparers = val_batch_preparers
        self.val_batch_preparer_names = list(val_batch_preparers.keys())

        # For logging / reproducibility
        if getattr(self.config_dict.task, "experiment_dir", None) is not None:
            self.save_dir = self.config_dict.task.experiment_dir

        n_params = sum(p.numel() for p in self.critic.parameters())
        n_trainable = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        self.config_dict.num_model_params = n_params
        self.config_dict.num_trainable_params = n_trainable

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.orchestrator(batch)

    # -------------------- training / validation --------------------

    def training_step(self, batch, batch_idx):
        prepared = self.train_batch_preparer(batch, device=self.device)
        losses = self(prepared)
        total = losses["total"]

        # Extract batch size from DNA tensor
        batch_size = prepared["dna"].size(0)

        # Log scalar losses with explicit batch_size
        self.log("train/loss_total", total, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, val in losses.items():
            if name == "total":
                continue
            self.log(f"train/{name}", val, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return total

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Loop over all validation batch preparers (each with different mask config)
        # This allows us to evaluate on multiple mask rate bins per validation batch
        for mask_name, val_batch_preparer in self.val_batch_preparers.items():
            # Prepare batch with this specific mask configuration
            prepared = val_batch_preparer(batch, device=self.device)
            losses = self(prepared)
            total = losses["total"]

            # Extract batch size from DNA tensor
            batch_size = prepared["dna"].size(0)

            # Log scalar losses with mask-specific prefix
            self.log(
                f"val_{mask_name}/loss_total",
                total,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            for loss_name, loss_val in losses.items():
                if loss_name == "total":
                    continue
                self.log(
                    f"val_{mask_name}/{loss_name}",
                    loss_val,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )

        # Return first mask's total loss (arbitrary choice for Lightning's internal tracking)
        first_mask_name = self.val_batch_preparer_names[0]
        first_prepared = self.val_batch_preparers[first_mask_name](batch, device=self.device)
        first_losses = self(first_prepared)
        return first_losses["total"]

    def on_train_epoch_start(self):
        """Called at the start of each training epoch to update dataset."""
        self.subsample_nonpeaks_for_epoch()

    def subsample_nonpeaks_for_epoch(self):
        """
        Update the training dataset's nonpeak sampling for the current epoch.
        """
        try:
            train_dataloader = self.trainer.train_dataloader
            dataset = train_dataloader.dataset
            if hasattr(dataset, "subsample_nonpeaks_for_epoch"):
                dataset.subsample_nonpeaks_for_epoch(self.current_epoch, verbose=True)
                self.print(f"Subsampled nonpeaks for epoch {self.current_epoch}")
        except Exception as e:
            raise Exception(f"Error subsampling nonpeaks for epoch {self.current_epoch}: {e}")

    # -------------------- optimizers --------------------

    def configure_optimizers(self):
        # Basic Adam optimizer with cosine or constant LR, driven by config_dict.lightning
        lr = float(getattr(self.config_dict.lightning, "base_lr", 1e-3))
        weight_decay = float(getattr(self.config_dict.lightning, "weight_decay", 0.0))

        optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        max_epochs = int(getattr(self.config_dict.lightning, "max_epochs", 100))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss_total",
            },
        }
