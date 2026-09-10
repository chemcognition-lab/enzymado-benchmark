"""Regression implementations for standard PyTorch benchmark models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from enzymado.models.common import (
    batch_to_input,
    make_optimizer,
    maybe_resume_torch_wrapper,
    mean_losses,
    prepare_inputs,
    restore_torch_training_state,
)
from enzymado.models.nn_blocks import LinearModel, MLPModel
from enzymado.training.torch_wrapper_base import BaseStandardWrapper


class LinearRegressorImplementation(BaseStandardWrapper):
    """Implement full-batch linear regression for the public wrapper."""

    def __init__(
        self,
        lr=0.01,
        epochs=200,
        optimizer_name="adamw",
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        super().__init__(
            lr,
            epochs,
            early_stopping_patience,
            early_stopping_min_delta,
            device,
        )
        self.optimizer_name = optimizer_name

    def fit(
        self,
        x,
        y,
        mask=None,
        x_val=None,
        y_val=None,
        mask_val=None,
        checkpoint_path=None,
    ):
        x, x_val = self._fit_and_apply_scaler(x, mask, x_val)
        x_t, mask_t, input_dim = prepare_inputs(x, mask, self.device)
        y_t = (
            torch.from_numpy(np.asarray(y, dtype=np.float32))
            .unsqueeze(1)
            .to(self.device)
        )
        self.model = LinearModel(input_dim, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.MSELoss()

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            y_val_t = (
                torch.from_numpy(np.asarray(y_val, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            val_payload = (x_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(x_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                x_val_t, mask_val_t, y_val_t = val_payload
                return criterion(self._forward(x_val_t, mask_val_t), y_val_t).item()

        self._run_training(
            train_epoch,
            val_loss if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer,
        )
        return self

    def predict(self, x, mask=None):
        x = self._apply_scaler(x)
        x_t, mask_t, _ = prepare_inputs(x, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(x_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(
        self,
        dataloader,
        batch_kind: str,
        val_dataloader=None,
        checkpoint_path=None,
    ):
        first_batch = next(iter(dataloader))
        x_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = x_t.shape[2]
        self.model = LinearModel(input_dim, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(x_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_t, mask_t, y_np = batch_to_input(
                        batch,
                        batch_kind,
                        self.device,
                    )
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(
                        float(criterion(self._forward(x_t, mask_t), y_t).item())
                    )
            return mean_losses(losses)

        self._run_training(
            train_epoch,
            val_loss if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer,
            dataloader,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                x_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(x_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


class MLPRegressorImplementation(BaseStandardWrapper):
    """Implement minibatch MLP regression for the public wrapper."""

    def __init__(
        self,
        hidden_dims=(2048,),
        lr=1e-3,
        epochs=100,
        optimizer_name="adamw",
        weight_decay=0.0,
        dropout=0.0,
        batch_size=256,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        super().__init__(
            lr,
            epochs,
            early_stopping_patience,
            early_stopping_min_delta,
            device,
        )
        self.hidden_dims = hidden_dims
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size

    def fit(
        self,
        x,
        y,
        mask=None,
        x_val=None,
        y_val=None,
        mask_val=None,
        checkpoint_path=None,
    ):
        x, x_val = self._fit_and_apply_scaler(x, mask, x_val)
        x_t, mask_t, input_dim = prepare_inputs(x, mask, self.device)
        y_t = (
            torch.from_numpy(np.asarray(y, dtype=np.float32))
            .unsqueeze(1)
            .to(self.device)
        )
        self.model = MLPModel(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(
            self.model.parameters(),
            self.optimizer_name,
            self.lr,
            self.weight_decay,
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.MSELoss()

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            y_val_t = (
                torch.from_numpy(np.asarray(y_val, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            val_payload = (x_val_t, mask_val_t, y_val_t)

        n = x_t.shape[0]

        def train_epoch():
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = x_t[idx]
                mb = mask_t[idx] if mask_t is not None else None
                yb = y_t[idx]
                optimizer.zero_grad()
                loss = criterion(self._forward(xb, mb), yb)
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            x_val_t, mask_val_t, y_val_t = val_payload
            nv = x_val_t.shape[0]
            losses = []
            with torch.no_grad():
                for start in range(0, nv, self.batch_size):
                    stop = min(start + self.batch_size, nv)
                    xb = x_val_t[start:stop]
                    mb = mask_val_t[start:stop] if mask_val_t is not None else None
                    yb = y_val_t[start:stop]
                    losses.append(float(criterion(self._forward(xb, mb), yb).item()))
            return mean_losses(losses)

        self._run_training(
            train_epoch,
            val_loss if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer,
        )
        return self

    def predict(self, x, mask=None):
        x = self._apply_scaler(x)
        x_t, mask_t, _ = prepare_inputs(x, mask, self.device)
        self.model.eval()
        n = x_t.shape[0]
        chunks = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                xb = x_t[start:stop]
                mb = mask_t[start:stop] if mask_t is not None else None
                chunks.append(self._forward(xb, mb).squeeze(1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)

    def fit_loader(
        self,
        dataloader,
        batch_kind: str,
        val_dataloader=None,
        checkpoint_path=None,
    ):
        first_batch = next(iter(dataloader))
        x_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = x_t.shape[2]
        self.model = MLPModel(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        optimizer = make_optimizer(
            self.model.parameters(),
            self.optimizer_name,
            self.lr,
            self.weight_decay,
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(x_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_t, mask_t, y_np = batch_to_input(
                        batch,
                        batch_kind,
                        self.device,
                    )
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(
                        float(criterion(self._forward(x_t, mask_t), y_t).item())
                    )
            return mean_losses(losses)

        self._run_training(
            train_epoch,
            val_loss if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer,
            dataloader,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                x_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(x_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


__all__ = ["LinearRegressorImplementation", "MLPRegressorImplementation"]
