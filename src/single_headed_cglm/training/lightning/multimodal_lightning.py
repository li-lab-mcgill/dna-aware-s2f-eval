"""
Lightning module for cgLM pretraining with BatchPreparer + Orchestrator pattern.
"""
import os
import sys
import json

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from ml_collections import ConfigDict

from torchmetrics.classification import Accuracy
from torchmetrics import MeanMetric

from typing import Dict, Any, Optional


class LightningModel(pl.LightningModule):
    """
    Lightning module for cgLM pretraining using BatchPreparer + Orchestrator pattern.

    Parameters
    ----------
    config_dict : ConfigDict
        Configuration dictionary
    orchestrator : nn.Module
        cgLMPretrainingOrchestrator that handles forward pass and loss
    train_batch_preparer : BatchPreparer
        Prepares training batches (applies masking)
    val_batch_preparer : BatchPreparer (optional)
        Single validation batch preparer. If not provided, uses train_batch_preparer.
    """

    def __init__(
        self,
        config_dict: ConfigDict,
        orchestrator: nn.Module,
        train_batch_preparer,
        val_batch_preparer=None,
    ):
        super(LightningModel, self).__init__()

        self.orchestrator = orchestrator
        self.train_batch_preparer = train_batch_preparer
        self.val_batch_preparer = val_batch_preparer

        # Get model and loss_fn from orchestrator for metrics
        self.model = orchestrator.model
        self.loss_fn = orchestrator.loss_fn

        # record all config
        self.config_dict = config_dict

        # All model related things are going to be saved here
        if self.config_dict.task.experiment_dir is not None:
            self.save_dir = self.config_dict.task.experiment_dir

        # Print number of model parameters
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._print("Number of parameters: %.2fM" % (self.n_params / 1e6,))
        self.config_dict.num_model_params = self.n_params

        """ Metrics to be tracked """
        ### train metrics
        # Global accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=4, ignore_index=self.loss_fn.ignore_index)
        # Peak-only accuracy for training
        self.train_accuracy_peaks = Accuracy(task="multiclass", num_classes=4, ignore_index=self.loss_fn.ignore_index)
        # Peak-only cross entropy for training
        self.train_ce_peaks = MeanMetric()

        ### validation metrics
        # Global accuracy
        self.val_accuracy = Accuracy(task="multiclass", num_classes=4, ignore_index=self.loss_fn.ignore_index)
        # Peak-only accuracy for validation
        self.val_accuracy_peaks = Accuracy(task="multiclass", num_classes=4, ignore_index=self.loss_fn.ignore_index)
        # Nonpeak-only accuracy for validation
        self.val_accuracy_nonpeaks = Accuracy(task="multiclass", num_classes=4, ignore_index=self.loss_fn.ignore_index)
        # Peak-only cross entropy for validation
        self.val_ce_peaks = MeanMetric()
        # Nonpeak-only cross entropy for validation
        self.val_ce_nonpeaks = MeanMetric()

        # accumulators for training and validation step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.training_mae_loss_outputs = []
        self.validation_mae_loss_outputs = []

    def forward(self, x):
        """Forward through the model (for inference)."""
        return self.model(x)

    """
    Training related functions
    """

    def training_loss_logging(self, losses):
        # Overall combined loss logging
        combined_loss = losses["total"]
        self.log(
            "train_loss",
            combined_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_CE",
            losses["L_dna"],
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        # Prepare batch using batch preparer
        prepared = self.train_batch_preparer(batch, device=self.device)

        # Forward through orchestrator
        losses = self.orchestrator(prepared)

        # Store the scalar losses for epoch-level metrics
        self.training_step_outputs.append(losses["total"].detach().cpu())
        self.training_mae_loss_outputs.append(losses["L_dna"].detach().cpu())

        # --- Log losses ---
        self.training_loss_logging(losses)

        # --- Compute metrics (use DNA branch for accuracy) ---
        # Get the predictions and targets for accuracy computation
        dna = prepared["dna"]
        pred_tracks = prepared.get("pred_tracks")
        exp_tracks = prepared.get("exp_tracks")
        mask = prepared["mask"]
        region_types = prepared.get("region_types")

        # Build data matrix for loss preprocessing
        tracks = pred_tracks if pred_tracks is not None else exp_tracks
        B, L_track, T = tracks.shape
        L_dna = dna.shape[1]
        if L_dna > L_track:
            offset = (L_dna - L_track) // 2
            dna_cropped = dna[:, offset:offset + L_track, :]
        else:
            dna_cropped = dna
        data_matrix = torch.cat([dna_cropped, tracks], dim=-1)

        # Build model input and get predictions
        masked_data = data_matrix.masked_fill(mask > 0, 0.0)
        model_input = torch.stack([masked_data, mask], dim=1)
        y_pred = self.model(model_input)

        # Get DNA labels
        dna_labels, _ = self.loss_fn.preprocess_data(data_matrix, mask)

        # Global accuracy (all samples)
        logits_flat = y_pred.reshape(-1, 4)
        labels_flat = dna_labels.reshape(-1)
        self.train_accuracy.update(logits_flat, labels_flat)

        # Peak-only accuracy and CE
        if region_types is not None:
            peak_mask = torch.tensor([rt == 'peak' for rt in region_types], device=y_pred.device)
            if peak_mask.any():
                peak_y_pred = y_pred[peak_mask]
                peak_dna_labels = dna_labels[peak_mask]
                peak_logits_flat = peak_y_pred.reshape(-1, 4)
                peak_labels_flat = peak_dna_labels.reshape(-1)
                self.train_accuracy_peaks.update(peak_logits_flat, peak_labels_flat)
                peak_ce = F.cross_entropy(peak_logits_flat, peak_labels_flat,
                                        ignore_index=self.loss_fn.ignore_index, reduction='mean')
                self.train_ce_peaks.update(peak_ce)

        return losses["total"]

    def on_train_epoch_start(self):
        """Called at the start of each training epoch to update dataset"""
        self.subsample_nonpeaks_for_epoch()

    def subsample_nonpeaks_for_epoch(self):
        """
        Update the training dataset's nonpeak sampling for the current epoch.
        This allows for dynamic resampling of negative examples during training.
        """
        try:
            # Get the training dataloader and its dataset
            train_dataloader = self.trainer.train_dataloader
            dataset = train_dataloader.dataset

            # Call the dataset's subsampling function with current epoch
            dataset.subsample_nonpeaks_for_epoch(self.current_epoch, verbose=True)

            self._print(f"Subsampled nonpeaks for epoch {self.current_epoch}")

        except Exception as e:
            raise Exception(f"Error subsampling nonpeaks for epoch {self.current_epoch}: {e}")

    def on_train_epoch_end(self):
        # Gather losses from all processes if in distributed mode
        if self.trainer.world_size > 1:
            gathered_outputs = self.all_gather(torch.stack(self.training_step_outputs))
            gathered_mae_loss = self.all_gather(torch.stack(self.training_mae_loss_outputs))
            train_loss = gathered_outputs.mean()
            train_mae_loss = gathered_mae_loss.mean()
        else:
            train_loss = torch.stack(self.training_step_outputs).mean()
            train_mae_loss = torch.stack(self.training_mae_loss_outputs).mean()

        # Compute metrics
        train_accuracy = self.train_accuracy.compute()
        train_accuracy_peaks = self.train_accuracy_peaks.compute()
        train_ce_peaks = self.train_ce_peaks.compute()

        # Log metrics
        self.log("train_accuracy", train_accuracy, sync_dist=True)
        self.log("train_accuracy_peaks", train_accuracy_peaks, sync_dist=True)
        self.log("train_ce_peaks", train_ce_peaks, sync_dist=True)

        # Print metrics in an organized way
        self._print(f"\n=== TRAINING METRICS (Epoch {self.current_epoch}) ===")
        self._print(f"Combined Loss: {train_loss:.4f}")
        self._print(f"DNA CE Loss: {train_mae_loss:.4f}")
        self._print(f"Accuracy (Global): {train_accuracy:.4f}")
        self._print(f"Accuracy (Peaks): {train_accuracy_peaks:.4f}")
        self._print(f"CE (Peaks): {train_ce_peaks:.4f}")

        # Reset metrics
        self.train_accuracy.reset()
        self.train_accuracy_peaks.reset()
        self.train_ce_peaks.reset()

        # Free memory
        self.training_step_outputs.clear()
        self.training_mae_loss_outputs.clear()

    """
    Validation related functions
    """

    def validation_loss_logging(self, losses):
        # Overall combined loss logging
        combined_loss = losses["total"]
        self.log(
            "val_loss",
            combined_loss,
            logger=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_CE",
            losses["L_dna"],
            logger=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Also log as val/L_dna for checkpoint compatibility
        self.log(
            "val/L_dna",
            losses["L_dna"],
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        # Use val_batch_preparer if provided, otherwise fall back to train_batch_preparer
        batch_preparer = self.val_batch_preparer if self.val_batch_preparer else self.train_batch_preparer

        # Prepare batch using batch preparer
        prepared = batch_preparer(batch, device=self.device)

        # Forward through orchestrator
        losses = self.orchestrator(prepared)

        # Store the scalar losses for validation metrics
        self.validation_step_outputs.append(losses["total"].detach().cpu())
        self.validation_mae_loss_outputs.append(losses["L_dna"].detach().cpu())

        # --- Log losses ---
        self.validation_loss_logging(losses)

        # --- Compute metrics ---
        dna = prepared["dna"]
        pred_tracks = prepared.get("pred_tracks")
        exp_tracks = prepared.get("exp_tracks")
        mask = prepared["mask"]
        region_types = prepared.get("region_types")

        # Build data matrix for loss preprocessing
        tracks = pred_tracks if pred_tracks is not None else exp_tracks
        B, L_track, T = tracks.shape
        L_dna = dna.shape[1]
        if L_dna > L_track:
            offset = (L_dna - L_track) // 2
            dna_cropped = dna[:, offset:offset + L_track, :]
        else:
            dna_cropped = dna
        data_matrix = torch.cat([dna_cropped, tracks], dim=-1)

        # Build model input and get predictions
        masked_data = data_matrix.masked_fill(mask > 0, 0.0)
        model_input = torch.stack([masked_data, mask], dim=1)
        y_pred = self.model(model_input)

        # Get DNA labels
        dna_labels, _ = self.loss_fn.preprocess_data(data_matrix, mask)

        # Global accuracy (all samples)
        logits_flat = y_pred.reshape(-1, 4)
        labels_flat = dna_labels.reshape(-1)
        self.val_accuracy.update(logits_flat, labels_flat)

        # Create masks for peak and nonpeak samples
        if region_types is not None:
            peak_mask = torch.tensor([rt == 'peak' for rt in region_types], device=y_pred.device)
            nonpeak_mask = torch.tensor([rt == 'nonpeak' for rt in region_types], device=y_pred.device)

            # Peak-only accuracy and CE
            if peak_mask.any():
                peak_y_pred = y_pred[peak_mask]
                peak_dna_labels = dna_labels[peak_mask]
                peak_logits_flat = peak_y_pred.reshape(-1, 4)
                peak_labels_flat = peak_dna_labels.reshape(-1)
                self.val_accuracy_peaks.update(peak_logits_flat, peak_labels_flat)
                peak_ce = F.cross_entropy(peak_logits_flat, peak_labels_flat,
                                        ignore_index=self.loss_fn.ignore_index, reduction='mean')
                self.val_ce_peaks.update(peak_ce)

            # Nonpeak-only accuracy and CE
            if nonpeak_mask.any():
                nonpeak_y_pred = y_pred[nonpeak_mask]
                nonpeak_dna_labels = dna_labels[nonpeak_mask]
                nonpeak_logits_flat = nonpeak_y_pred.reshape(-1, 4)
                nonpeak_labels_flat = nonpeak_dna_labels.reshape(-1)
                self.val_accuracy_nonpeaks.update(nonpeak_logits_flat, nonpeak_labels_flat)
                nonpeak_ce = F.cross_entropy(nonpeak_logits_flat, nonpeak_labels_flat,
                                           ignore_index=self.loss_fn.ignore_index, reduction='mean')
                self.val_ce_nonpeaks.update(nonpeak_ce)

        return losses["total"]

    def on_validation_epoch_end(self):
        # Safety check - skip if no outputs collected
        if not self.validation_step_outputs:
            self._print("Warning: No validation outputs collected, skipping metrics computation")
            return

        # Gather losses from all processes if in distributed mode
        if self.trainer.world_size > 1:
            gathered_outputs = self.all_gather(torch.stack(self.validation_step_outputs))
            gathered_mae_loss = self.all_gather(torch.stack(self.validation_mae_loss_outputs))
            val_loss = gathered_outputs.mean()
            val_mae_loss = gathered_mae_loss.mean()
        else:
            val_loss = torch.stack(self.validation_step_outputs).mean()
            val_mae_loss = torch.stack(self.validation_mae_loss_outputs).mean()

        # Compute metrics
        val_accuracy = self.val_accuracy.compute()
        val_accuracy_peaks = self.val_accuracy_peaks.compute()
        val_accuracy_nonpeaks = self.val_accuracy_nonpeaks.compute()
        val_ce_peaks = self.val_ce_peaks.compute()
        val_ce_nonpeaks = self.val_ce_nonpeaks.compute()

        # Log metrics
        self.log("val_accuracy", val_accuracy, sync_dist=True)
        self.log("val_accuracy_peaks", val_accuracy_peaks, sync_dist=True)
        self.log("val_accuracy_nonpeaks", val_accuracy_nonpeaks, sync_dist=True)
        self.log("val_ce_peaks", val_ce_peaks, sync_dist=True)
        self.log("val_ce_nonpeaks", val_ce_nonpeaks, sync_dist=True)

        # Print metrics in an organized way
        self._print(f"\n=== VALIDATION METRICS ===")
        self._print(f"Combined Loss: {val_loss:.4f}")
        self._print(f"DNA CE Loss: {val_mae_loss:.4f}")
        self._print(f"Accuracy (Global): {val_accuracy:.4f}")
        self._print(f"Accuracy (Peaks): {val_accuracy_peaks:.4f}")
        self._print(f"Accuracy (Nonpeaks): {val_accuracy_nonpeaks:.4f}")
        self._print(f"CE (Peaks): {val_ce_peaks:.4f}")
        self._print(f"CE (Nonpeaks): {val_ce_nonpeaks:.4f}")

        # Reset metrics
        self.val_accuracy.reset()
        self.val_accuracy_peaks.reset()
        self.val_accuracy_nonpeaks.reset()
        self.val_ce_peaks.reset()
        self.val_ce_nonpeaks.reset()

        # Free memory
        self.validation_step_outputs.clear()
        self.validation_mae_loss_outputs.clear()

    def configure_optimizers(self):
        # List to hold optimizers
        optimizers = []

        # Model parameters only (no CLIP model)
        optimizers.append(optim.AdamW(self.model.parameters(),
            lr=float(self.config_dict.lightning.base_lr),
            weight_decay=float(self.config_dict.lightning.weight_decay)
        ))

        # List to hold the schedulers
        schedulers = []

        if self.config_dict.lightning.warmup_type:
            if self.config_dict.lightning.warmup_type == "linear":
                def warmup_schedule(epoch):
                    if epoch < self.config_dict.lightning.warmup_epochs:
                        return float(epoch) / float(max(1, self.config_dict.lightning.warmup_epochs))
                    return 1.0

            warmup_scheduler = LambdaLR(optimizers[-1], lr_lambda=warmup_schedule)
            schedulers.append({
                'scheduler': warmup_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'warmup_scheduler'
            })

        if self.config_dict.lightning.cooldown_type:
            if self.config_dict.lightning.cooldown_type == "cosine":
                cooldown_scheduler = CosineAnnealingLR(optimizers[-1],
                        T_max=self.config_dict.lightning.cooldown_epochs,
                        eta_min=float(self.config_dict.lightning.min_lr))

            schedulers.append({
                'scheduler': cooldown_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'cooldown_scheduler'
            })

        if len(schedulers) > 0:
            return optimizers, schedulers
        else:
            return optimizers

    def train_on_dataset(
        self,
        train_dataloader,
        val_dataloader,
    ):
        """
        Train model.

        Returns:
            pl.Trainer object
        """

        # Set torch precision
        torch.set_float32_matmul_precision("high")

        # Get the process rank for DDP awareness
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        is_main_process = global_rank == 0

        # Logger - ONLY create on main process
        if is_main_process:
            # Set logger dir
            logger = CSVLogger(self.save_dir)
            self.logger_dir = logger.log_dir
            os.makedirs(self.logger_dir, exist_ok=True)

            # Save config_dict - ONLY on main process
            config_save_path = osp.join(self.logger_dir, "config_dict.json")
            config_json = json.loads(self.config_dict.to_json_best_effort())
            with open(config_save_path, "w") as f:
                json.dump(config_json, f, indent=4)
        else:
            # For non-main processes no logger is needed
            logger = None

        # Synchronize all processes to ensure main process has created directories
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Create callbacks - add CE-based early stopping and checkpoints
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", filename="best_val_loss-{epoch}-{step}"),
            ModelCheckpoint(monitor="val_CE", mode="min", filename="best_val_CE-{epoch}-{step}"),
            ModelCheckpoint(monitor="val_ce_peaks", mode="min", filename="best_val_CE_peaks-{epoch}-{step}"),
            ModelCheckpoint(monitor="val_ce_nonpeaks", mode="min", filename="best_val_CE_nonpeaks-{epoch}-{step}"),
            LearningRateMonitor(logging_interval="epoch"),
            # Early stopping can now use peak CE as it's more meaningful
            EarlyStopping(monitor="val_ce_peaks", mode="min", patience=self.config_dict.lightning.es_patience),
        ]

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.config_dict.lightning.max_epochs,
            accelerator=self.config_dict.lightning.accelerator,
            devices=self.config_dict.lightning.devices,
            strategy=self.config_dict.lightning.strategy,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.save_dir,
            val_check_interval=self.config_dict.lightning.val_check_interval,
            gradient_clip_val=self.config_dict.lightning.gradient_clip_val,
            deterministic=self.config_dict.lightning.deterministic,
            precision=self.config_dict.lightning.precision,
        )

        # First validation pass
        trainer.validate(model=self, dataloaders=val_dataloader)

        # Training
        trainer.fit(model=self,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

        return trainer

    def _print(self, *args, **kwargs):
        """Print only on the main process or if trainer is not initialized yet."""
        try:
            # Try to access the trainer and check if we're the global zero process
            if self.trainer.is_global_zero:
                print(*args, **kwargs)
        except (RuntimeError, AttributeError):
            # If trainer is not initialized yet, just print
            # This catches the "not attached to a Trainer" error
            print(*args, **kwargs)
