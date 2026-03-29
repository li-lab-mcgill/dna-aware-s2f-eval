import os
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from ml_collections import ConfigDict


class LightningWrapper(pl.LightningModule):
    """
    Lightning module for DNA-free DAE training (iteration-2).

    - Uses a BatchPreparer to build (dna, exp_tracks, pred_tracks, critic_input_mask, critic_output_mask, ...) batches.
    - Wraps an orchestrator which returns (total_loss, stats).
    - Optimizes DAE parameters; bias generator and critic are kept frozen (inside the orchestrator).
    - Supports multiple validation batch preparers (dict) for evaluating on different mask configurations.
    """

    def __init__(
        self,
        config_dict: ConfigDict,
        orchestrator: torch.nn.Module,
        train_batch_preparer,
        val_batch_preparers: Dict[str, Any],
    ):
        super().__init__()
        self.config_dict = config_dict
        self.orchestrator = orchestrator
        self.train_batch_preparer = train_batch_preparer
        self.val_batch_preparers = val_batch_preparers
        self.val_batch_preparer_names = list(val_batch_preparers.keys())

        if getattr(self.config_dict.task, "experiment_dir", None) is not None:
            self.save_dir = self.config_dict.task.experiment_dir

        # Parameter report:
        # - Trainable parameters should primarily live in orchestrator.dae.
        # - orchestrator.bias_generator and critic (inside adapter) should be frozen.
        total_params = sum(p.numel() for p in self.orchestrator.parameters())
        trainable_params = sum(p.numel() for p in self.orchestrator.parameters() if p.requires_grad)
        self.config_dict.num_model_params = total_params
        self.config_dict.num_trainable_params = trainable_params
        self.config_dict.num_frozen_params = total_params - trainable_params

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

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        total_loss, stats = self.orchestrator(batch)
        out: Dict[str, torch.Tensor] = {"loss_total": total_loss}
        out.update(stats)
        return out

    # -------------------- training / validation --------------------

    def training_step(self, batch, batch_idx):
        prepared = self.train_batch_preparer(batch, device=self.device)

        total_loss, stats = self.orchestrator(prepared)

        batch_size = prepared["dna"].size(0)

        # Log main loss
        self.log(
            "train/loss_total",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log detailed stats (fid/*, critic/*, etc.)
        for name, value in stats.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() != 0:
                continue
            if name == "loss_total":
                continue
            self.log(
                f"train/{name}",
                value,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return total_loss

    def _log_validation_stats(
        self,
        prefix: str,
        total_loss: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        batch_size: int,
        *,
        prog_bar_loss: bool = False,
    ) -> None:
        self.log(
            f"{prefix}/loss_total",
            total_loss,
            prog_bar=prog_bar_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

        for name, value in stats.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() != 0:
                continue
            if name == "loss_total":
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Loop over all validation batch preparers (each with different mask config)
        # This allows us to evaluate on multiple mask rate bins per validation batch
        first_total_loss = None
        first_mask_name = self.val_batch_preparer_names[0]
        for mask_name, val_batch_preparer in self.val_batch_preparers.items():
            # Prepare batch with this specific mask configuration
            prepared = val_batch_preparer(batch, device=self.device)
            # Validation uses the batch-provided masks; no override or mode split.
            invco_total_loss, invco_stats = self.orchestrator(
                prepared,
                mode="val",
            )

            # Extract batch size from DNA tensor
            batch_size = prepared["dna"].size(0)

            # Default logging keeps existing checkpoint monitor keys.
            self._log_validation_stats(
                f"val_{mask_name}",
                invco_total_loss,
                invco_stats,
                batch_size,
                prog_bar_loss=True,
            )

            if mask_name == first_mask_name:
                first_total_loss = invco_total_loss

        # Return first mask's total loss (arbitrary choice for Lightning's internal tracking)
        return first_total_loss

    # -------------------- optimizers --------------------

    def configure_optimizers(self):
        # Optimize only DAE parameters (bias generator and critic are frozen).
        dae_params = self.orchestrator.dae.parameters()

        lr = float(getattr(self.config_dict.lightning, "base_lr", 1e-3))
        weight_decay = float(getattr(self.config_dict.lightning, "weight_decay", 0.0))

        optimizer = torch.optim.AdamW(dae_params, lr=lr, weight_decay=weight_decay)

        max_epochs = int(getattr(self.config_dict.lightning, "max_epochs", 100))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )

        # Match scheduler monitor metric to EarlyStopping: mid-range mask bin fidelity loss if available.
        if "40to60" in self.val_batch_preparers:
            monitor_metric = "val_40to60/fid/total"
        else:
            first_mask = self.val_batch_preparer_names[0]
            monitor_metric = f"val_{first_mask}/fid/total"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": monitor_metric,
            },
        }


__all__ = ["LightningWrapper"]
