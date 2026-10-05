"""Shared state and input handling for standard PyTorch wrappers."""

import torch

from enzymado.models.common import (
    DEVICE,
    apply_feature_scaler,
    fit_feature_scaler,
    masked_mean_pool,
    run_wrapper_training,
    validation_objective,
)
import enzymado.models.training_protocol


class BaseStandardWrapper:
    """Store training state shared by pooled and residue-aware wrappers.

    ``validation_metric`` names the score early stopping selects on. Subclasses
    that leave it ``None`` keep selecting on validation loss, which is what the
    EC and residue routes do: evaluating a full metric every epoch is too
    expensive there. This is a protocol choice, not a guarantee that loss and
    ranking metrics agree.
    """

    validation_metric: str | None = None

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
        # Mirrored onto the instance so a resumed checkpoint records which
        # criterion selected it, even if the class default later changes.
        self.validation_metric_ = self.validation_metric

    def _forward(self, x_t, mask_t):
        out = self.model(x_t)
        if x_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def _fit_and_apply_scaler(self, x, mask, x_val=None, unscaled_columns=None):
        self.x_mean_, self.x_scale_ = fit_feature_scaler(x, mask)
        if unscaled_columns is not None:
            # Ligand descriptors already lie in [0, 1]. A column that is nearly
            # constant in train would turn a test value into a huge z-score.
            start, stop = unscaled_columns
            self.x_mean_[start:stop] = 0.0
            self.x_scale_[start:stop] = 1.0
        x = apply_feature_scaler(x, self.x_mean_, self.x_scale_)
        if x_val is not None:
            x_val = apply_feature_scaler(x_val, self.x_mean_, self.x_scale_)
        return x, x_val

    def _apply_scaler(self, x):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            return apply_feature_scaler(x, self.x_mean_, self.x_scale_)
        return x

    def _selection_objective(self, logits_t, y_t):
        """Return the negated selection metric for concatenated validation output.

        Callers must pass the whole validation split at once: AUPRC and Kendall
        tau are not averages, so scoring per batch and taking the mean would not
        give the value the leaderboard reports.
        """
        enzymado.models.training_protocol.check_finite_tensors(
            [logits_t], "validation logits"
        )
        scores = (
            torch.sigmoid(logits_t)
            if self.validation_metric == "auprc"
            else logits_t
        )
        return validation_objective(
            y_t.reshape(-1).detach().cpu().numpy(),
            scores.reshape(-1).detach().cpu().numpy(),
            self.validation_metric,
        )

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
