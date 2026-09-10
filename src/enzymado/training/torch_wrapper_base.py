"""Shared state and input handling for standard PyTorch wrappers."""

import torch

from enzymado.models.common import (
    DEVICE,
    apply_feature_scaler,
    fit_feature_scaler,
    masked_mean_pool,
    run_wrapper_training,
)


class BaseStandardWrapper:
    """Store training state shared by pooled and residue-aware wrappers."""

    def __init__(
        self,
        lr,
        epochs,
        early_stopping_patience,
        early_stopping_min_delta,
        device,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.x_mean_ = None
        self.x_scale_ = None

    def _forward(self, x_t, mask_t):
        out = self.model(x_t)
        if x_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def _fit_and_apply_scaler(self, x, mask, x_val=None):
        self.x_mean_, self.x_scale_ = fit_feature_scaler(x, mask)
        x = apply_feature_scaler(x, self.x_mean_, self.x_scale_)
        if x_val is not None:
            x_val = apply_feature_scaler(x_val, self.x_mean_, self.x_scale_)
        return x, x_val

    def _apply_scaler(self, x):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            return apply_feature_scaler(x, self.x_mean_, self.x_scale_)
        return x

    def _run_training(
        self,
        train_epoch_fn,
        val_loss_fn,
        start_epoch,
        checkpoint_path,
        optimizer,
        dataloader=None,
    ) -> None:
        run_wrapper_training(
            self,
            train_epoch_fn,
            val_loss_fn,
            start_epoch,
            checkpoint_path,
            optimizer=optimizer,
            dataloader=dataloader,
        )


__all__ = ["BaseStandardWrapper"]
