"""Torch wrappers for standard pooled and residue-aware classification/regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.common import (
    DEVICE,
    apply_feature_scaler,
    batch_to_input,
    fit_feature_scaler,
    make_optimizer,
    masked_mean_pool,
    maybe_resume_torch_wrapper,
    prepare_inputs,
    run_wrapper_training,
    validation_score,
)
from models.nn_blocks import LinearModel, MLPModel


def _score_logits(y_true, logits: torch.Tensor, metric_name: str) -> float:
    """Score validation logits with the task's checkpoint metric."""
    y_true = (
        y_true.detach().cpu().numpy()
        if torch.is_tensor(y_true)
        else np.asarray(y_true)
    )
    match metric_name:
        case "auprc":
            pred = torch.sigmoid(logits).cpu().numpy()
        case "kendall_tau":
            pred = logits.cpu().numpy()
        case _:
            raise ValueError(f"Unsupported validation metric: {metric_name}.")
    # Targets arrive as (N, 1) from fit() and (N,) from the loader path, while
    # logits are always (N, 1). Flatten both so evaluate_predictions() sees
    # matching shapes; a mismatch there silently yields zero metrics.
    return validation_score(np.ravel(y_true), np.ravel(pred), metric_name)


def _score_loader(wrapper, dataloader, batch_kind: str) -> float:
    """Score one standard-wrapper validation loader in a single pass."""
    y_chunks = []
    pred_chunks = []
    wrapper.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_t, mask_t, y_true = batch_to_input(
                batch,
                batch_kind,
                wrapper.device,
            )
            logits = wrapper._forward(x_t, mask_t)
            y_chunks.append(np.asarray(y_true))
            pred_chunks.append(logits)
    return _score_logits(
        np.concatenate(y_chunks),
        torch.cat(pred_chunks),
        wrapper.validation_metric_,
    )


class _BaseStandardWrapper:
    """General wrapper class"""

    validation_metric: str

    def __init__(self, lr, epochs, early_stopping_patience, early_stopping_min_delta, device):
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_score_ = None
        self.validation_metric_ = self.validation_metric
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.x_mean_ = None
        self.x_scale_ = None

    def _forward(self, X_t, mask_t):
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = masked_mean_pool(out, mask_t)
        return out

    def _fit_and_apply_scaler(self, X, mask, x_val=None):
        self.x_mean_, self.x_scale_ = fit_feature_scaler(X, mask)
        X = apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        if x_val is not None:
            x_val = apply_feature_scaler(x_val, self.x_mean_, self.x_scale_)
        return X, x_val

    def _apply_scaler(self, X):
        if self.x_mean_ is not None and self.x_scale_ is not None:
            return apply_feature_scaler(X, self.x_mean_, self.x_scale_)
        return X

    def _run_training(self, train_epoch_fn, val_score_fn, start_epoch, checkpoint_path):
        run_wrapper_training(
            self,
            train_epoch_fn,
            val_score_fn,
            start_epoch,
            checkpoint_path,
        )


class LinearRegressorTorch(_BaseStandardWrapper):
    """Linear regressor baseline for 2D pooled or 3D residue-aware inputs."""

    validation_metric = "kendall_tau"

    def __init__(self, lr=0.01, epochs=200, optimizer_name="adamw", early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        super().__init__(lr, epochs, early_stopping_patience, early_stopping_min_delta, device)
        self.optimizer_name = optimizer_name

    def fit(self, X, y, mask=None, x_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        X, x_val = self._fit_and_apply_scaler(X, mask, x_val)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = LinearModel(input_dim, output_dim=1).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.MSELoss()

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            val_payload = (x_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_score():
            self.model.eval()
            with torch.no_grad():
                x_val_t, mask_val_t, y_val_t = val_payload
                logits = self._forward(x_val_t, mask_val_t)
            return _score_logits(
                y_val_t,
                logits,
                self.validation_metric_,
            )

        self._run_training(
            train_epoch,
            val_score if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
        )
        return self

    def predict(self, X, mask=None):
        X = self._apply_scaler(X)
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
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_score():
            return _score_loader(self, val_dataloader, batch_kind)

        self._run_training(
            train_epoch,
            val_score if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
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


class LinearClassifierTorch(_BaseStandardWrapper):
    """Linear classifier baseline for pooled or residue-aware embedding inputs."""

    validation_metric = "auprc"

    def __init__(self, lr=0.001, epochs=200, optimizer_name="adamw", early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        super().__init__(lr, epochs, early_stopping_patience, early_stopping_min_delta, device)
        self.optimizer_name = optimizer_name
        self.classes_ = None

    def fit(self, X, y, mask=None, x_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        X, x_val = self._fit_and_apply_scaler(X, mask, x_val)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(self.device)

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            else:
                y_val_t = torch.tensor([label_map[v] for v in np.asarray(y_val)], dtype=torch.long).to(self.device)
            val_payload = (x_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()

        def val_score():
            self.model.eval()
            with torch.no_grad():
                x_val_t, mask_val_t, y_val_t = val_payload
                logits = self._forward(x_val_t, mask_val_t)
            return _score_logits(
                y_val_t,
                logits,
                self.validation_metric_,
            )

        self._run_training(
            train_epoch,
            val_score if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
        )
        return self

    def predict(self, X, mask=None):
        X = self._apply_scaler(X)
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
        X = self._apply_scaler(X)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        unique_classes = set()
        input_dim = None
        for batch in dataloader:
            X_t, _, _ = batch_to_input(batch, batch_kind, self.device)
            if input_dim is None:
                input_dim = X_t.shape[2]
            unique_classes.update(np.asarray(batch.target, dtype=np.float32).ravel().tolist())
        self.classes_ = np.array(sorted(unique_classes), dtype=np.float32)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
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

        def val_score():
            return _score_loader(self, val_dataloader, batch_kind)

        self._run_training(
            train_epoch,
            val_score if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
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

    def predict_scores_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        """Return positive-class probabilities for loader-based inference."""
        if len(self.classes_) != 2:
            raise ValueError("Continuous scores are only defined for binary classification.")
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                scores.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        return np.concatenate(scores)


class MLPRegressorTorch(_BaseStandardWrapper):
    """MLP regressor for pooled or residue-aware embedding inputs."""

    validation_metric = "kendall_tau"

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
        super().__init__(lr, epochs, early_stopping_patience, early_stopping_min_delta, device)
        self.hidden_dims = hidden_dims
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size

    def fit(self, X, y, mask=None, x_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        X, x_val = self._fit_and_apply_scaler(X, mask, x_val)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.MSELoss()

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            val_payload = (x_val_t, mask_val_t, y_val_t)

        n = X_t.shape[0]

        def train_epoch():
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xb = X_t[idx]
                mb = mask_t[idx] if mask_t is not None else None
                yb = y_t[idx]
                optimizer.zero_grad()
                loss = criterion(self._forward(Xb, mb), yb)
                loss.backward()
                optimizer.step()

        def val_score():
            self.model.eval()
            x_val_t, mask_val_t, y_val_t = val_payload
            nv = x_val_t.shape[0]
            logits = []
            with torch.no_grad():
                for start in range(0, nv, self.batch_size):
                    stop = min(start + self.batch_size, nv)
                    Xb = x_val_t[start:stop]
                    mb = mask_val_t[start:stop] if mask_val_t is not None else None
                    logits.append(self._forward(Xb, mb))
            return _score_logits(
                y_val_t,
                torch.cat(logits),
                self.validation_metric_,
            )

        self._run_training(
            train_epoch,
            val_score if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
        )
        return self

    def predict(self, X, mask=None):
        X = self._apply_scaler(X)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        n = X_t.shape[0]
        chunks = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                Xb = X_t[start:stop]
                mb = mask_t[start:stop] if mask_t is not None else None
                chunks.append(self._forward(Xb, mb).squeeze(1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        first_batch = next(iter(dataloader))
        X_t, _, _ = batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_score():
            return _score_loader(self, val_dataloader, batch_kind)

        self._run_training(
            train_epoch,
            val_score if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
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


class MLPClassifierTorch(_BaseStandardWrapper):
    """MLP classifier for pooled or residue-aware embedding inputs."""

    validation_metric = "auprc"

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
        super().__init__(lr, epochs, early_stopping_patience, early_stopping_min_delta, device)
        self.hidden_dims = hidden_dims
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.classes_ = None

    def fit(self, X, y, mask=None, x_val=None, y_val=None, mask_val=None, checkpoint_path=None):
        X, x_val = self._fit_and_apply_scaler(X, mask, x_val)
        X_t, mask_t, input_dim = prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(self.device)

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).unsqueeze(1).to(self.device)
            else:
                y_val_t = torch.tensor([label_map[v] for v in np.asarray(y_val)], dtype=torch.long).to(self.device)
            val_payload = (x_val_t, mask_val_t, y_val_t)

        n = X_t.shape[0]

        def train_epoch():
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xb = X_t[idx]
                mb = mask_t[idx] if mask_t is not None else None
                yb = y_t[idx]
                optimizer.zero_grad()
                loss = criterion(self._forward(Xb, mb), yb)
                loss.backward()
                optimizer.step()

        def val_score():
            self.model.eval()
            x_val_t, mask_val_t, y_val_t = val_payload
            nv = x_val_t.shape[0]
            logits = []
            with torch.no_grad():
                for start in range(0, nv, self.batch_size):
                    stop = min(start + self.batch_size, nv)
                    Xb = x_val_t[start:stop]
                    mb = mask_val_t[start:stop] if mask_val_t is not None else None
                    logits.append(self._forward(Xb, mb))
            return _score_logits(
                y_val_t,
                torch.cat(logits),
                self.validation_metric_,
            )

        self._run_training(
            train_epoch,
            val_score if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
        )
        return self

    def _chunked_logits(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        n = X_t.shape[0]
        chunks = []
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            Xb = X_t[start:stop]
            mb = mask_t[start:stop] if mask_t is not None else None
            chunks.append(self._forward(Xb, mb))
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 1), device=self.device)

    def predict(self, X, mask=None):
        X = self._apply_scaler(X)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._chunked_logits(X_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X, mask=None):
        X = self._apply_scaler(X)
        X_t, mask_t, _ = prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._chunked_logits(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        unique_classes = set()
        input_dim = None
        for batch in dataloader:
            X_t, _, _ = batch_to_input(batch, batch_kind, self.device)
            if input_dim is None:
                input_dim = X_t.shape[2]
            unique_classes.update(np.asarray(batch.target, dtype=np.float32).ravel().tolist())
        self.classes_ = np.array(sorted(unique_classes), dtype=np.float32)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        self.model = MLPModel(input_dim, self.hidden_dims, output_dim, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
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

        def val_score():
            return _score_loader(self, val_dataloader, batch_kind)

        self._run_training(
            train_epoch,
            val_score if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
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

    def predict_scores_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        """Return positive-class probabilities for loader-based inference."""
        if len(self.classes_) != 2:
            raise ValueError("Continuous scores are only defined for binary classification.")
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                scores.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        return np.concatenate(scores)


__all__ = [
    "LinearRegressorTorch",
    "LinearClassifierTorch",
    "MLPRegressorTorch",
    "MLPClassifierTorch",
]
