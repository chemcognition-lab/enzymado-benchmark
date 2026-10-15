"""Hierarchical EC classification with pooled PyTorch embeddings."""

from __future__ import annotations

import collections.abc
from pathlib import Path
import typing

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

import enzymado.models.common
import enzymado.models.nn_blocks
import enzymado.models.training_protocol
import enzymado.types

type FloatArray = npt.NDArray[np.float32]
type BoolArray = npt.NDArray[np.bool_]
type ECTargets = dict[str, FloatArray]
type ECMasks = dict[str, BoolArray]
type ECValidationPayload = tuple[FloatArray, ECTargets, ECMasks]
type ECMinibatch = tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]


class ECMultiHeadClassifierImplementation:
    """Shared-backbone classifier for hierarchical EC targets on pooled inputs.

    Checkpoints are selected on validation AUPRC pooled over the four EC levels,
    the leaderboard estimator, accumulated batch by batch in a fixed histogram.
    """

    validation_metric = "auprc"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (2048,),
        lr: float = 1e-3,
        epochs: int = 100,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        batch_size: int = 256,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.validation_metric_ = self.validation_metric
        self.device = (
            torch.device(device) if device else enzymado.models.common.DEVICE
        )
        self.model: enzymado.models.nn_blocks.MultiHeadMLPModel | None = None
        self.output_dims: dict[str, int] | None = None
        self.best_epoch_: int | None = None
        self.best_val_loss_: float | None = None
        self.current_epoch_ = 0
        self.best_model_state_: enzymado.models.common.ModelState | None = None
        self.x_mean_: FloatArray | None = None
        self.x_scale_: FloatArray | None = None

    @staticmethod
    def _targets_to_tensors(
        y: ECTargets,
        target_mask: ECMasks,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        y_t = {
            level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(
                device
            )
            for level_name, target in y.items()
        }
        target_mask_t = {
            level_name: torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
            for level_name, mask in target_mask.items()
        }
        return y_t, target_mask_t

    def _forward(self, x_t: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_t.ndim != 2:
            raise ValueError(
                "ECMultiHeadClassifierTorch expects pooled 2D inputs, "
                f"got shape {tuple(x_t.shape)}"
            )
        if self.model is None:
            raise RuntimeError("EC classifier model has not been initialized.")
        return self.model(x_t)

    def _iter_minibatches(
        self,
        x: FloatArray,
        targets: ECTargets,
        target_masks: ECMasks,
    ) -> collections.abc.Iterator[ECMinibatch]:
        """Yield device tensors for aligned EC target minibatches."""
        for start in range(0, x.shape[0], self.batch_size):
            stop = min(start + self.batch_size, x.shape[0])
            x_t = torch.from_numpy(x[start:stop]).to(self.device)
            y_t, target_mask_t = self._targets_to_tensors(
                {
                    level_name: target[start:stop]
                    for level_name, target in targets.items()
                },
                {
                    level_name: mask[start:stop]
                    for level_name, mask in target_masks.items()
                },
                self.device,
            )
            yield x_t, y_t, target_mask_t

    def _train_epoch(
        self,
        x: FloatArray,
        targets: ECTargets,
        target_masks: ECMasks,
        optimizer: torch.optim.Optimizer,
        criterion: nn.BCEWithLogitsLoss,
    ) -> None:
        """Train the hierarchical classifier for one epoch."""
        if self.model is None:
            raise RuntimeError("EC classifier model has not been initialized.")
        self.model.train()
        for x_t, y_t, target_mask_t in self._iter_minibatches(
            x,
            targets,
            target_masks,
        ):
            logits_by_level = self._forward(x_t)
            loss = enzymado.models.common.compute_ec_batch_loss(
                logits_by_level,
                y_t,
                target_mask_t,
                criterion,
            )
            if loss is None:
                continue
            optimizer.zero_grad()
            enzymado.models.training_protocol.backward_checked(loss)
            optimizer.step()

    def _validation_objective(self, payload: ECValidationPayload) -> float:
        """Return negative validation AUPRC pooled over every EC level."""
        if self.model is None:
            raise RuntimeError("EC classifier model has not been initialized.")
        self.model.eval()
        auprc = enzymado.models.common.StreamingAUPRC(self.device)
        x_val, y_val, target_mask_val = payload
        with torch.no_grad():
            for x_t, y_t, target_mask_t in self._iter_minibatches(
                x_val,
                y_val,
                target_mask_val,
            ):
                for level_name, logits in self._forward(x_t).items():
                    auprc.update(
                        torch.sigmoid(logits.float()),
                        y_t[level_name],
                        target_mask_t[level_name],
                    )
        return -auprc.value()

    def fit(
        self,
        x: enzymado.types.Array,
        y: dict[str, enzymado.types.Array],
        target_mask: dict[str, enzymado.types.Array] | None,
        x_val: enzymado.types.Array | None = None,
        y_val: dict[str, enzymado.types.Array] | None = None,
        target_mask_val: dict[str, enzymado.types.Array] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> typing.Self:
        if target_mask is None:
            raise ValueError("ECMultiHeadClassifierTorch.fit requires target_mask.")

        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"x must be pooled 2D, got shape {x_np.shape}")
        self.x_mean_, self.x_scale_ = enzymado.models.common.fit_feature_scaler(x_np)
        x_np = enzymado.models.common.apply_feature_scaler(
            x_np,
            self.x_mean_,
            self.x_scale_,
        )
        if x_val is not None:
            x_val = enzymado.models.common.apply_feature_scaler(
                np.asarray(x_val, dtype=np.float32),
                self.x_mean_,
                self.x_scale_,
            )
        input_dim = x_np.shape[1]

        targets: ECTargets = {
            level_name: np.asarray(target, dtype=np.float32)
            for level_name, target in y.items()
        }
        target_masks: ECMasks = {
            level_name: np.asarray(mask, dtype=bool)
            for level_name, mask in target_mask.items()
        }
        self.output_dims = {
            level_name: int(target.shape[1])
            for level_name, target in targets.items()
        }
        self.model = enzymado.models.nn_blocks.MultiHeadMLPModel(
            input_dim,
            self.hidden_dims,
            self.output_dims,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = enzymado.models.common.maybe_resume_torch_wrapper(
            self,
            checkpoint_path,
        )
        optimizer = typing.cast(
            torch.optim.Optimizer,
            enzymado.models.common.make_optimizer(
                self.model.parameters(),
                self.optimizer_name,
                self.lr,
                self.weight_decay,
            ),
        )
        enzymado.models.common.restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        val_payload: ECValidationPayload | None = None
        if x_val is not None and y_val is not None and target_mask_val is not None:
            val_payload = (
                np.asarray(x_val, dtype=np.float32),
                {
                    level_name: np.asarray(target, dtype=np.float32)
                    for level_name, target in y_val.items()
                },
                {
                    level_name: np.asarray(mask_arr, dtype=bool)
                    for level_name, mask_arr in target_mask_val.items()
                },
            )

        def train_epoch() -> None:
            self._train_epoch(
                x_np,
                targets,
                target_masks,
                optimizer,
                criterion,
            )

        validation_loss: collections.abc.Callable[[], float] | None = None
        if val_payload is not None:

            def val_loss() -> float:
                return self._validation_objective(val_payload)

            validation_loss = val_loss

        enzymado.models.common.run_wrapper_training(
            self,
            train_epoch,
            validation_loss,
            start_epoch,
            checkpoint_path,
            optimizer=optimizer,
        )
        return self

    def _predict(
        self,
        x: enzymado.types.Array,
        *,
        proba: bool,
    ) -> dict[str, FloatArray]:
        if self.output_dims is None:
            raise ValueError(
                "ECMultiHeadClassifierTorch must be fit before calling predict()."
            )

        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"x must be pooled 2D, got shape {x_np.shape}")
        if self.x_mean_ is not None and self.x_scale_ is not None:
            x_np = enzymado.models.common.apply_feature_scaler(
                x_np,
                self.x_mean_,
                self.x_scale_,
            )
        preds_by_level: dict[str, list[FloatArray]] = {
            level_name: [] for level_name in self.output_dims
        }

        if self.model is None:
            raise RuntimeError("EC classifier model has not been initialized.")
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x_np.shape[0], self.batch_size):
                stop = min(start + self.batch_size, x_np.shape[0])
                x_t = torch.from_numpy(x_np[start:stop]).to(self.device)
                logits_by_level = self._forward(x_t)
                for level_name, logits in logits_by_level.items():
                    preds_by_level[level_name].append(
                        torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
                        if proba
                        else (torch.sigmoid(logits) >= 0.5)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

        return {
            level_name: np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, self.output_dims[level_name]), dtype=np.float32)
            for level_name, chunks in preds_by_level.items()
        }

    def predict(self, x: enzymado.types.Array) -> dict[str, FloatArray]:
        return self._predict(x, proba=False)

    def predict_proba(self, x: enzymado.types.Array) -> dict[str, FloatArray]:
        """Return per-level sigmoid probabilities instead of hard labels."""
        return self._predict(x, proba=True)


__all__ = ["ECMultiHeadClassifierImplementation"]
