"""Torch wrappers for standard pooled and residue-aware classification/regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.common import (
    DEVICE,
    batch_to_input,
    make_torch_checkpoint_callback,
    masked_mean_pool,
    maybe_resume_torch_wrapper,
    prepare_inputs,
    run_training_with_early_stopping,
)
from models.nn_blocks import LinearModel, MLPModel


def _fit_feature_scaler(X, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple per-feature standardizer for 2D or padded 3D inputs."""
    X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 2:
        observed = X_np
    elif X_np.ndim == 3:
        if mask is None:
            observed = X_np.reshape(-1, X_np.shape[-1])
        else:
            observed = X_np[np.asarray(mask, dtype=bool)]
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {X_np.shape}")

    mean = observed.mean(axis=0).astype(np.float32, copy=False)
    scale = observed.std(axis=0).astype(np.float32, copy=False)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def _apply_feature_scaler(X, mean: np.ndarray, scale: np.ndarray):
    """Apply a fitted per-feature standardizer."""
    return (np.asarray(X, dtype=np.float32) - mean) / scale


class LinearRegressorTorch:
    """Linear regressor baseline for 2D pooled or 3D residue-aware inputs."""

    def __init__(self, lr=0.01, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
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

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        self.x_mean_, self.x_scale_ = _fit_feature_scaler(X, mask)
        X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        if X_val is not None:
            X_val = _apply_feature_scaler(X_val, self.x_mean_, self.x_scale_)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = LinearModel(input_dim, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = prepare_inputs(X_val, mask_val, self.device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            val_payload = (X_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                X_val_t, mask_val_t, y_val_t = val_payload
                return criterion(self._forward(X_val_t, mask_val_t), y_val_t).item()

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        first_batch = next(iter(dataloader))
        X_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = LinearModel(input_dim, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(X_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


class LinearClassifierTorch:
    """Linear classifier baseline for pooled or residue-aware embedding inputs."""

    def __init__(self, lr=0.001, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.classes_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.x_mean_ = None
        self.x_scale_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        self.x_mean_, self.x_scale_ = _fit_feature_scaler(X, mask)
        X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        if X_val is not None:
            X_val = _apply_feature_scaler(X_val, self.x_mean_, self.x_scale_)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(self.device)

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = prepare_inputs(X_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            else:
                y_val_t = torch.tensor([label_map[v] for v in np.asarray(y_val)], dtype=torch.long).to(self.device)
            val_payload = (X_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                X_val_t, mask_val_t, y_val_t = val_payload
                return criterion(self._forward(X_val_t, mask_val_t), y_val_t).item()

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        all_y = np.concatenate([np.asarray(batch.target, dtype=np.float32) for batch in dataloader])
        self.classes_ = np.unique(all_y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        first_batch = next(iter(dataloader))
        X_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                if len(self.classes_) == 2:
                    p = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


class MLPRegressorTorch:
    """MLP regressor for pooled or residue-aware embedding inputs."""

    def __init__(self, hidden_dims=(512, 256), lr=1e-3, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        self.hidden_dims = hidden_dims
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

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        self.x_mean_, self.x_scale_ = _fit_feature_scaler(X, mask)
        X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        if X_val is not None:
            X_val = _apply_feature_scaler(X_val, self.x_mean_, self.x_scale_)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = prepare_inputs(X_val, mask_val, self.device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            val_payload = (X_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                X_val_t, mask_val_t, y_val_t = val_payload
                return criterion(self._forward(X_val_t, mask_val_t), y_val_t).item()

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        first_batch = next(iter(dataloader))
        X_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(X_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


class MLPClassifierTorch:
    """MLP classifier for pooled or residue-aware embedding inputs."""

    def __init__(self, hidden_dims=(512, 256), lr=1e-3, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.classes_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.x_mean_ = None
        self.x_scale_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        self.x_mean_, self.x_scale_ = _fit_feature_scaler(X, mask)
        X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        if X_val is not None:
            X_val = _apply_feature_scaler(X_val, self.x_mean_, self.x_scale_)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(self.device)

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = prepare_inputs(X_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            else:
                y_val_t = torch.tensor([label_map[v] for v in np.asarray(y_val)], dtype=torch.long).to(self.device)
            val_payload = (X_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                X_val_t, mask_val_t, y_val_t = val_payload
                return criterion(self._forward(X_val_t, mask_val_t), y_val_t).item()

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X, mask=None):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X = _apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        all_y = np.concatenate([np.asarray(batch.target, dtype=np.float32) for batch in dataloader])
        self.classes_ = np.unique(all_y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        first_batch = next(iter(dataloader))
        X_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                if len(self.classes_) == 2:
                    p = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


# Backward-compatible aliases while we migrate away from the SGD naming.
SGDRegressorTorch = LinearRegressorTorch
SGDClassifierTorch = LinearClassifierTorch


__all__ = [
    "LinearRegressorTorch",
    "LinearClassifierTorch",
    "MLPRegressorTorch",
    "MLPClassifierTorch",
    "SGDRegressorTorch",
    "SGDClassifierTorch",
]
