from typing import Dict, Any, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler # For Automatic Mixed Precision
from torch.amp import autocast
import os
from pathlib import Path
from tqdm import tqdm
from box import Box

from ..utils.registry import (
    MODEL_REGISTRY,
    METRIC_REGISTRY,
    CRITERION_REGISTRY,
)
from ..metrics.manager import MetricManager
from ..utils.logger import BaseLogger
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.seeds import seed_setup

class Trainer:
    """

    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[Any],  # Can be _LRScheduler or ReduceLROnPlateau
            train_loader: DataLoader,
            valid_loader: DataLoader,
            test_loader: DataLoader,
            criterion: torch.nn.Module,
            config: Box,
            logger: BaseLogger,
            metrics_manager: MetricManager,  # Pass MetricManager instead of individual metrics
            device: torch.device,
    ):
        """

        :param model:
        :param optimizer:
        :param scheduler:
        :param train_loader:
        :param valid_loader:
        :param test_loader:
        :param criterion:
        :param config:
        :param logger:
        :param metrics_manager:
        :param device:
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.metrics_manager = metrics_manager
        self.device = device

        self.model.to(self.device)
        self.criterion.to(self.device)
        # Setup GradScaler for Automatic Mixed Precision if enabled
        self.use_amp = self.config.trainer.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) is enabled.")
        else:
            self.logger.info("Automatic Mixed Precision (AMP) is disabled.")

        self.global_step = 0
        self.start_epoch = 0
        self.best_score = -float('inf') if self.config.trainer.get("monitor_mode", "max") == "max" else float('inf')
        self.monitor_metric = self.config.trainer.get("monitor", "val/loss")
        self.monitor_mode = self.config.trainer.get("monitor_mode", "max")
        self.grad_clip_value = self.config.trainer.get("grad_clip", None)
        self.epochs = self.config.trainer.epochs
        self.save_dir = Path(self.config.trainer.save_dir)  # save_dir is updated in main.py
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.debug_mode = self.config.trainer.get("debug", False)
        if self.debug_mode:
            self.logger.info("Debug mode is enabled. Will only process one batch per loader.")

        # Resume from checkpoint if specified
        if self.config.trainer.get("resume_from"):
            self._resume_checkpoint(self.config.trainer.resume_from)

        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Monitor metric: {self.monitor_metric}, Mode: {self.monitor_mode}")
        self.is_log_train = self.config.trainer.get("is_log_train", False)

    def _resume_checkpoint(self, resume_path: Union[str, Path]) -> None:
        """
        Resumes training from a saved checkpoint.
        :param resume_path:
        :return:
        """

        try:
            checkpoint = load_checkpoint(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_score = checkpoint.get('best_score', self.best_score)
            self.logger.info(f"Resumed training from epoch {self.start_epoch} with global_step {self.global_step}")
        except FileNotFoundError:
            self.logger.warning(f"Resume checkpoint not found at {resume_path}. Starting training from scratch.")
        except Exception as e:
            self.logger.error(f"Error resuming from checkpoint {resume_path}: {e}. Starting training from scratch.")

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Any, Tensor]:
        """
        Moves batch data to device and handles input formatting.
        """
        if isinstance(batch['inputs'], dict):
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
        else:
            inputs = batch['inputs'].to(self.device)
        targets = batch['targets'].to(self.device)

        # Ensure targets are 1D for CrossEntropyLoss
        if targets.dim() > 1:
            targets = targets.squeeze()
        return inputs, targets

    def _forward(self, inputs: Any) -> Dict[str, Tensor]:
        outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

        if isinstance(outputs, dict):
            return outputs
        elif isinstance(outputs, (list, tuple)):
            return {'logits': outputs[-1], 'features': outputs[0] if len(outputs) > 1 else None}
        else:
            return {'logits': outputs}

    def _evaluation_step(self, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """

        :param batch:
        :return:
        """
        inputs, targets = self._prepare_batch(batch)
        with autocast(str(self.device), enabled=self.use_amp):
            outputs_dict = self._forward(inputs)
            logits = outputs_dict['logits']
            loss = self.criterion(logits, targets)
        return loss, outputs_dict

    def _run_evaluation_loop(self, loader: DataLoader, desc: str) -> Tuple[float, int]:
        """
        Runs the evaluation loop (validation/test).
        """
        total_loss = 0.0
        batch_idx = -1

        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for batch_idx, batch in enumerate(pbar):
                loss, outputs_dict = self._evaluation_step(batch)
                total_loss += loss.item()
                self.metrics_manager.update(outputs_dict, batch)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if self.debug_mode:
                    break
        return total_loss, batch_idx + 1

    def train_step(self, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """

        :param batch:
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()

        inputs, targets = self._prepare_batch(batch)

        with autocast(str(self.device), enabled=self.use_amp):
            outputs_dict = self._forward(inputs)
            logits = outputs_dict['logits']

            # Loss calculation
            loss = self.criterion(logits, targets)

        if self.scaler:
            self.scaler.scale(loss).backward()
            if self.grad_clip_value:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            self.optimizer.step()

        self.global_step += 1

        # Log training loss
        self.logger.log_metrics({'train/loss': loss.item()}, step=self.global_step)

        return loss.detach(), outputs_dict  # Return detached loss and model outputs

    def train_epoch(self, epoch: int) -> None:
        """
        Runs a full training epoch.
        :param epoch:
        :return:
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        self.metrics_manager.reset()  # Reset metrics at the start of each epoch for training metrics (if needed)

        batch_idx = 0
        for batch_idx, batch in enumerate(pbar):
            loss, outputs = self.train_step(batch)
            total_loss += loss.item()

            # Update metrics with training data
            if self.is_log_train:
                self.metrics_manager.update(outputs, batch)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if self.debug_mode:  # Only process one batch in debug mode
                break

        avg_loss = total_loss / (batch_idx + 1)
        self.logger.info(f"Epoch {epoch} Training - Average Loss: {avg_loss:.4f}")
        if self.is_log_train:
            train_metrics = self.metrics_manager.compute()
            self.logger.log_metrics({f"train/{k}": v for k,v in train_metrics.items()}, step=epoch)

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs a full validation epoch.
        :param epoch:
        :return:
        """
        self.model.eval()
        self.metrics_manager.reset()  # Reset metrics for validation
        
        total_val_loss, num_batches = self._run_evaluation_loop(self.valid_loader, f"Epoch {epoch} Validation")
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        
        val_scores = self.metrics_manager.compute()
        val_scores['val/loss'] = avg_val_loss  # Add validation loss to scores

        # Log validation metrics
        self.logger.log_metrics(val_scores, step=epoch)
        self.logger.info(f"Epoch {epoch} Validation - Results: {val_scores}")
        return val_scores

    def test_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs a full test epoch.
        :param epoch:
        :return:
        """
        self.model.eval()
        self.metrics_manager.reset()

        total_test_loss, num_batches = self._run_evaluation_loop(self.test_loader, "Final Test")
        avg_test_loss = total_test_loss / num_batches if num_batches > 0 else 0.0

        test_scores = self.metrics_manager.compute()
        test_scores['test/loss'] = avg_test_loss  # Add test loss to scores

        # Log test metrics
        self.logger.log_metrics(test_scores, step=epoch)
        self.logger.info(f"Final Test - Results: {test_scores}")
        return test_scores

    def _save_current_state(self, epoch: int, is_best: bool = False,
                            file_name: str = "last.pth") -> None:
        """
        Helper to save the current training state.
        :param epoch:
        :param is_best:
        :param file_name:
        :return:
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config.to_dict()  # Save config for reproducibility
        }
        if self.scheduler:
            # Check if scheduler has state_dict before saving
            if hasattr(self.scheduler, 'state_dict') and callable(getattr(self.scheduler, 'state_dict')):
                state['scheduler_state_dict'] = self.scheduler.state_dict()

        save_checkpoint(state, self.save_dir, file_name=file_name, is_best=is_best)

    def emergency_save(self, epoch: int, file_name: str = "emergency_save.pth"):
        """

        :param epoch:
        :param file_name:
        :return:
        """
        self._save_current_state(epoch, file_name=file_name)

    def run(self) -> None:
        """
        Runs the main training loop for the specified number of epochs.
        """
        self.logger.info(f"Starting training for {self.epochs - self.start_epoch} epochs (total {self.epochs}).")

        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(f"--- Epoch {epoch}/{self.epochs - 1} ---")
            self.train_epoch(epoch)
            val_scores = self.validate_epoch(epoch)

            # Update learning rate scheduler (if not ReduceLROnPlateau)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # Update learning rate scheduler (if ReduceLROnPlateau)
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Ensure the monitor metric is available in val_scores
                if self.monitor_metric not in val_scores:
                    self.logger.warning(
                        f"Monitor metric '{self.monitor_metric}' not found in validation scores. "
                        f"Scheduler {type(self.scheduler).__name__} will not step."
                    )
                else:
                    self.scheduler.step(val_scores[self.monitor_metric])

            # Check for best model and save checkpoint
            current_score = val_scores.get(self.monitor_metric, None)
            if current_score is None:
                self.logger.warning(f"Monitor metric '{self.monitor_metric}' not found in validation scores. "
                                    "Best model tracking and saving will be skipped for this epoch.")
                # Still save the last checkpoint
                # TODO current_score is None
                self._save_current_state(epoch, is_best=False, file_name="last.pth")
                continue

            is_best = False
            if self.monitor_mode == "max" and current_score > self.best_score:
                self.best_score = current_score
                is_best = True
                self.logger.info(f"New best score for {self.monitor_metric}: {self.best_score:.4f} at epoch {epoch}")
            elif self.monitor_mode == "min" and current_score < self.best_score:
                self.best_score = current_score
                is_best = True
                self.logger.info(f"New best score for {self.monitor_metric}: {self.best_score:.4f} at epoch {epoch}")

            self._save_current_state(epoch, is_best=is_best, file_name="last.pth")

        self.logger.info("Training finished.")

        if len(self.test_loader) != 0:
            test_scores = self.test_epoch(self.epochs)
            self.logger.info("Final test finished")