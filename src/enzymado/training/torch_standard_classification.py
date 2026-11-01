"""Classification implementations for standard PyTorch benchmark models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from enzymado.models import training_protocol
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


class LinearClassifierImplementation(BaseStandardWrapper):
    """Linear classifier baseline for pooled or residue-aware embedding inputs."""

    validation_metric = "auprc"

    def __init__(
        self,
        lr=0.001,
        epochs=200,
        optimizer_name="adamw",
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        super().__init__(
            lr, epochs, early_stopping_patience, early_stopping_min_delta, device
        )
        self.optimizer_name = optimizer_name
        self.classes_ = None

    def fit(
        self,
        x,
        y,
        mask=None,
        x_val=None,
        y_val=None,
        mask_val=None,
        checkpoint_path=None,
        unscaled_columns=None,
    ):
        x, x_val = self._fit_and_apply_scaler(x, mask, x_val, unscaled_columns)
        x_t, mask_t, input_dim = prepare_inputs(x, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr
        )
        restore_torch_training_state(self, optimizer)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(
                self.device
            )

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = (
                    torch.from_numpy(np.asarray(y_val, dtype=np.float32))
                    .unsqueeze(1)
                    .to(self.device)
                )
            else:
                y_val_t = torch.tensor(
                    [label_map[v] for v in np.asarray(y_val)], dtype=torch.long
                ).to(self.device)
            val_payload = (x_val_t, mask_val_t, y_val_t)

        def train_epoch():
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self._forward(x_t, mask_t), y_t)
            training_protocol.backward_checked(loss)
            optimizer.step()

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                x_val_t, mask_val_t, y_val_t = val_payload
                logits = self._forward(x_val_t, mask_val_t)
                if self.validation_metric is None or n_classes != 2:
                    return float(criterion(logits, y_val_t).item())
                return self._selection_objective(logits, y_val_t)

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
            logits = self._forward(x_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, x, mask=None):
        x = self._apply_scaler(x)
        x_t, mask_t, _ = prepare_inputs(x, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(x_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
        unique_classes = set()
        input_dim = None
        for batch in dataloader:
            x_t, _, _ = batch_to_input(batch, batch_kind, self.device)
            if input_dim is None:
                input_dim = x_t.shape[2]
            unique_classes.update(
                np.asarray(batch.target, dtype=np.float32).ravel().tolist()
            )
        self.classes_ = np.array(sorted(unique_classes), dtype=np.float32)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        self.model = LinearModel(input_dim, output_dim).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor(
                        [label_map[v] for v in y_np], dtype=torch.long
                    ).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._forward(x_t, mask_t), y_t)
                training_protocol.backward_checked(loss)
                optimizer.step()

        def val_loss():
            self.model.eval()
            use_metric = self.validation_metric is not None and n_classes == 2
            losses = []
            logit_chunks = []
            label_chunks = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor(
                            [label_map[v] for v in y_np], dtype=torch.long
                        ).to(self.device)
                    logits = self._forward(x_t, mask_t)
                    if use_metric:
                        logit_chunks.append(logits)
                        label_chunks.append(y_t)
                    else:
                        losses.append(float(criterion(logits, y_t).item()))
            if use_metric:
                return self._selection_objective(
                    torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)
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
                logits = self._forward(x_t, mask_t)
                if len(self.classes_) == 2:
                    p = (
                        (torch.sigmoid(logits).squeeze(1) >= 0.5)
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


class MLPClassifierImplementation(BaseStandardWrapper):
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
        super().__init__(
            lr, epochs, early_stopping_patience, early_stopping_min_delta, device
        )
        self.hidden_dims = hidden_dims
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.classes_ = None

    def fit(
        self,
        x,
        y,
        mask=None,
        x_val=None,
        y_val=None,
        mask_val=None,
        checkpoint_path=None,
        unscaled_columns=None,
    ):
        x, x_val = self._fit_and_apply_scaler(x, mask, x_val, unscaled_columns)
        x_t, mask_t, input_dim = prepare_inputs(x, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = MLPModel(
            input_dim, self.hidden_dims, output_dim, dropout=self.dropout
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay
        )
        restore_torch_training_state(self, optimizer)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(
                self.device
            )

        val_payload = None
        if x_val is not None and y_val is not None:
            x_val_t, mask_val_t, _ = prepare_inputs(x_val, mask_val, self.device)
            if n_classes == 2:
                y_val_t = (
                    torch.from_numpy(np.asarray(y_val, dtype=np.float32))
                    .unsqueeze(1)
                    .to(self.device)
                )
            else:
                y_val_t = torch.tensor(
                    [label_map[v] for v in np.asarray(y_val)], dtype=torch.long
                ).to(self.device)
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
                training_protocol.backward_checked(loss)
                optimizer.step()

        def val_loss():
            self.model.eval()
            x_val_t, mask_val_t, y_val_t = val_payload
            nv = x_val_t.shape[0]
            use_metric = self.validation_metric is not None and n_classes == 2
            losses = []
            chunks = []
            with torch.no_grad():
                for start in range(0, nv, self.batch_size):
                    stop = min(start + self.batch_size, nv)
                    xb = x_val_t[start:stop]
                    mb = mask_val_t[start:stop] if mask_val_t is not None else None
                    logits = self._forward(xb, mb)
                    if use_metric:
                        chunks.append(logits)
                    else:
                        losses.append(
                            float(criterion(logits, y_val_t[start:stop]).item())
                        )
            if use_metric:
                return self._selection_objective(torch.cat(chunks, dim=0), y_val_t)
            return mean_losses(losses)

        self._run_training(
            train_epoch,
            val_loss if val_payload is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer,
        )
        return self

    def _chunked_logits(
        self, x_t: torch.Tensor, mask_t: torch.Tensor | None
    ) -> torch.Tensor:
        n = x_t.shape[0]
        chunks = []
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            xb = x_t[start:stop]
            mb = mask_t[start:stop] if mask_t is not None else None
            chunks.append(self._forward(xb, mb))
        return (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.zeros((0, 1), device=self.device)
        )

    def predict(self, x, mask=None):
        x = self._apply_scaler(x)
        x_t, mask_t, _ = prepare_inputs(x, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._chunked_logits(x_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, x, mask=None):
        x = self._apply_scaler(x)
        x_t, mask_t, _ = prepare_inputs(x, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._chunked_logits(x_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
        unique_classes = set()
        input_dim = None
        for batch in dataloader:
            x_t, _, _ = batch_to_input(batch, batch_kind, self.device)
            if input_dim is None:
                input_dim = x_t.shape[2]
            unique_classes.update(
                np.asarray(batch.target, dtype=np.float32).ravel().tolist()
            )
        self.classes_ = np.array(sorted(unique_classes), dtype=np.float32)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        self.model = MLPModel(
            input_dim, self.hidden_dims, output_dim, dropout=self.dropout
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor(
                        [label_map[v] for v in y_np], dtype=torch.long
                    ).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._forward(x_t, mask_t), y_t)
                training_protocol.backward_checked(loss)
                optimizer.step()

        def val_loss():
            self.model.eval()
            use_metric = self.validation_metric is not None and n_classes == 2
            losses = []
            logit_chunks = []
            label_chunks = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_t, mask_t, y_np = batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor(
                            [label_map[v] for v in y_np], dtype=torch.long
                        ).to(self.device)
                    logits = self._forward(x_t, mask_t)
                    if use_metric:
                        logit_chunks.append(logits)
                        label_chunks.append(y_t)
                    else:
                        losses.append(float(criterion(logits, y_t).item()))
            if use_metric:
                return self._selection_objective(
                    torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)
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
                logits = self._forward(x_t, mask_t)
                if len(self.classes_) == 2:
                    p = (
                        (torch.sigmoid(logits).squeeze(1) >= 0.5)
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


__all__ = [
    "LinearClassifierImplementation",
    "MLPClassifierImplementation",
]
