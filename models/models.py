import copy

try:
    from xgboost import XGBRegressor, XGBClassifier
except ModuleNotFoundError:
    XGBRegressor = None
    XGBClassifier = None
import numpy as np
import torch
import torch.nn as nn


class SGDModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class SharedMLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Identity() if not layers else nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, X):
        return self.net(X)


class MultiHeadMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims):
        super().__init__()
        self.backbone = SharedMLPBackbone(input_dim, hidden_dims)
        self.heads = nn.ModuleDict({
            head_name: nn.Linear(self.backbone.output_dim, output_dim)
            for head_name, output_dim in output_dims.items()
        })
        self.output_dims = dict(output_dims)

    def forward(self, X):
        features = self.backbone(X)
        return {
            head_name: head(features)
            for head_name, head in self.heads.items()
        }


def _masked_mean_pool(out: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool (batch, seq_len, dim) → (batch, dim) using boolean mask."""
    if mask is None:
        return out.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()           # (batch, seq_len, 1)
    return (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_optimizer(parameters, optimizer_name: str, lr: float):
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer_name '{optimizer_name}'.")


def _pad_1d_array_list(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=dtype)
    max_length = max(array.shape[0] for array in arrays)
    batch = np.zeros((len(arrays), max_length), dtype=dtype)
    for index, array in enumerate(arrays):
        batch[index, : array.shape[0]] = array.astype(dtype, copy=False)
    return batch


def _batch_to_input(batch, batch_kind: str, device: torch.device):
    """Extract (X_t, mask_t, y_np) from a collated batch for residue-aware training.

    Supports protein_global and mutation_delta_global batch kinds.
    Returns X_t as 3D (batch, seq_len, emb_dim) tensor on device.
    """
    if batch_kind == "protein_global":
        X_t = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    elif batch_kind == "mutation_delta_global":
        X_t = torch.from_numpy(np.asarray(batch.protein_delta_emb, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    else:
        raise ValueError(f"fit_loader does not support batch_kind='{batch_kind}'")
    y_np = np.asarray(batch.target, dtype=np.float32)
    return X_t, mask_t, y_np


def _prepare_inputs(
    X: np.ndarray, mask: np.ndarray | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return (X_tensor, mask_tensor, input_dim) on the given device.

    Accepts:
    - 2D (n_samples, emb_dim)  — mask ignored
    - 3D (n_samples, seq_len, emb_dim) — mask shape (n_samples, seq_len)
    """
    X = np.asarray(X, dtype=np.float32)
    mask_t = None
    if X.ndim == 3:
        input_dim = X.shape[2]
        if mask is not None:
            mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
    elif X.ndim == 2:
        input_dim = X.shape[1]
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
    return torch.from_numpy(X).to(device), mask_t, input_dim


def _run_training_with_early_stopping(
    model: nn.Module,
    epochs: int,
    train_epoch_fn,
    val_loss_fn=None,
    patience: int | None = None,
    min_delta: float = 0.0,
) -> tuple[int | None, float | None]:
    if val_loss_fn is None or patience is None or patience <= 0:
        for _ in range(epochs):
            train_epoch_fn()
        return None, None

    best_state = copy.deepcopy(model.state_dict())
    best_epoch: int | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_epoch_fn()
        val_loss = float(val_loss_fn())
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            continue

        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_loss


class SGDRegressorTorch:
    """SGDModel wrapper for regression. Accepts 2D (n, emb) or 3D (n, seq, emb) + mask."""

    def __init__(self, lr=0.01, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)                      # (..., output_dim)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)   # (batch, output_dim)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = SGDModel(input_dim, output_dim=1).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = _prepare_inputs(X_val, mask_val, self.device)
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

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = SGDModel(input_dim, output_dim=1).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = _batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(X_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


class SGDClassifierTorch:
    """SGDModel wrapper for classification. Accepts 2D (n, emb) or 3D (n, seq, emb) + mask."""

    def __init__(self, lr=0.01, epochs=200, early_stopping_patience=None, early_stopping_min_delta=0.0, device=None):
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.classes_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = SGDModel(input_dim, output_dim).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss()
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_t = torch.tensor([label_map[v] for v in y], dtype=torch.long).to(self.device)

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = _prepare_inputs(X_val, mask_val, self.device)
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

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        else:
            return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        # Pre-pass to determine classes_
        all_y = np.concatenate([
            np.asarray(batch.target, dtype=np.float32)
            for batch in dataloader
        ])
        self.classes_ = np.unique(all_y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = SGDModel(input_dim, output_dim).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
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
                    X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = _batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                if len(self.classes_) == 2:
                    p = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


class MLPRegressorTorch:
    """MLPModel wrapper for regression. Accepts 2D (n, emb) or 3D (n, seq, emb) + mask."""

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

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_t, mask_val_t, _ = _prepare_inputs(X_val, mask_val, self.device)
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

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = _batch_to_input(batch, batch_kind, self.device)
                preds.append(self._forward(X_t, mask_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds)


class MLPClassifierTorch:
    """MLPModel wrapper for classification. Accepts 2D (n, emb) or 3D (n, seq, emb) + mask."""

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

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim).to(self.device)
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
            X_val_t, mask_val_t, _ = _prepare_inputs(X_val, mask_val, self.device)
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

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(X_t, mask_t)
        if len(self.classes_) == 2:
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return np.column_stack([1 - p, p])
        else:
            return torch.softmax(logits, dim=1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        all_y = np.concatenate([
            np.asarray(batch.target, dtype=np.float32) for batch in dataloader
        ])
        self.classes_ = np.unique(all_y)
        n_classes = len(self.classes_)
        output_dim = 1 if n_classes == 2 else n_classes
        label_map = {c: i for i, c in enumerate(self.classes_)}

        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
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
                    X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                    if n_classes == 2:
                        y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                    else:
                        y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                    losses.append(float(criterion(self._forward(X_t, mask_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _ = _batch_to_input(batch, batch_kind, self.device)
                logits = self._forward(X_t, mask_t)
                if len(self.classes_) == 2:
                    p = (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                else:
                    p = logits.argmax(dim=1).cpu().numpy()
                preds.append(self.classes_[p])
        return np.concatenate(preds)


class ECMultiHeadClassifierTorch:
    """Multi-label hierarchical EC classifier over pooled protein embeddings."""

    def __init__(
        self,
        hidden_dims=(),
        lr=1e-3,
        epochs=200,
        optimizer_name="adam",
        batch_size=1024,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.output_dims = None
        self.best_epoch_ = None
        self.best_val_loss_ = None

    @staticmethod
    def _targets_to_tensors(y, target_mask, device: torch.device):
        y_t = {
            level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)
            for level_name, target in y.items()
        }
        target_mask_t = {
            level_name: torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
            for level_name, mask in target_mask.items()
        }
        return y_t, target_mask_t

    @staticmethod
    def _compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion):
        loss_terms = []
        for level_name, logits in logits_by_level.items():
            level_loss = criterion(logits, y_t[level_name]).mean(dim=1)
            sample_mask = target_mask_t[level_name].float()
            denom = sample_mask.sum()
            if denom.item() == 0:
                continue
            loss_terms.append((level_loss * sample_mask).sum() / denom)
        if not loss_terms:
            return None
        return torch.stack(loss_terms).mean()

    def _forward(self, X_t: torch.Tensor):
        if X_t.ndim != 2:
            raise ValueError(f"ECMultiHeadClassifierTorch expects pooled 2D inputs, got shape {tuple(X_t.shape)}")
        return self.model(X_t)

    def fit(self, X, y, target_mask, X_val=None, y_val=None, target_mask_val=None):
        if target_mask is None:
            raise ValueError("ECMultiHeadClassifierTorch.fit requires target_mask.")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        input_dim = X_np.shape[1]

        self.output_dims = {
            level_name: int(np.asarray(target, dtype=np.float32).shape[1])
            for level_name, target in y.items()
        }
        self.model = MultiHeadMLPModel(input_dim, self.hidden_dims, self.output_dims).to(self.device)
        optimizer = _make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        val_payload = None
        if X_val is not None and y_val is not None and target_mask_val is not None:
            val_payload = (
                np.asarray(X_val, dtype=np.float32),
                {level_name: np.asarray(target, dtype=np.float32) for level_name, target in y_val.items()},
                {level_name: np.asarray(mask_arr, dtype=bool) for level_name, mask_arr in target_mask_val.items()},
            )

        def iter_minibatches(X_arr, y_arr, target_mask_arr):
            for start in range(0, X_arr.shape[0], self.batch_size):
                stop = min(start + self.batch_size, X_arr.shape[0])
                X_t = torch.from_numpy(X_arr[start:stop]).to(self.device)
                y_t, target_mask_t = self._targets_to_tensors(
                    {level_name: target[start:stop] for level_name, target in y_arr.items()},
                    {level_name: mask_values[start:stop] for level_name, mask_values in target_mask_arr.items()},
                    self.device,
                )
                yield X_t, y_t, target_mask_t

        def train_epoch():
            self.model.train()
            for X_t, y_t, target_mask_t in iter_minibatches(X_np, y, target_mask):
                logits_by_level = self._forward(X_t)
                loss = self._compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                X_val_np, y_val_np, target_mask_val_np = val_payload
                for X_t, y_t, target_mask_t in iter_minibatches(
                    X_val_np,
                    y_val_np,
                    target_mask_val_np,
                ):
                    logits_by_level = self._forward(X_t)
                    loss = self._compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                    if loss is not None:
                        losses.append(float(loss.item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        if self.output_dims is None:
            raise ValueError("ECMultiHeadClassifierTorch must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        preds_by_level = {level_name: [] for level_name in self.output_dims}

        self.model.eval()
        with torch.no_grad():
            for start in range(0, X_np.shape[0], self.batch_size):
                stop = min(start + self.batch_size, X_np.shape[0])
                X_t = torch.from_numpy(X_np[start:stop]).to(self.device)
                logits_by_level = self._forward(X_t)
                for level_name, logits in logits_by_level.items():
                    preds_by_level[level_name].append(
                        (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
                    )

        return {
            level_name: np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, self.output_dims[level_name]), dtype=np.float32)
            for level_name, chunks in preds_by_level.items()
        }


class XGBFlatECClassifier:
    """One binary XGBoost model per concatenated EC column.

    Unknown EC descendants are represented as NaN in `y` and excluded by the
    per-column `target_mask`.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        early_stopping_rounds=None,
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use XGBFlatECClassifier.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.models_: list[XGBClassifier | None] = []
        self.constant_predictions_: list[float | None] = []
        self.n_outputs_: int | None = None

    @staticmethod
    def _normalize_targets(
        y,
        target_mask=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(f"XGBFlatECClassifier expects 2D targets, got shape {y_np.shape}")

        inferred_mask = ~np.isnan(y_np)
        if target_mask is None:
            mask_np = inferred_mask
        else:
            mask_np = np.asarray(target_mask, dtype=bool)
            if mask_np.shape != y_np.shape:
                raise ValueError(
                    "target_mask shape must match y shape for XGBFlatECClassifier. "
                    f"Received mask={mask_np.shape}, y={y_np.shape}."
                )
            mask_np = mask_np & inferred_mask
        return y_np, mask_np

    def _make_model(self, use_early_stopping: bool) -> XGBClassifier:
        params = {
            "objective": "binary:logistic",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
        }
        if use_early_stopping and self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        return XGBClassifier(**params)

    def fit(self, X, y, target_mask=None, X_val=None, y_val=None, target_mask_val=None):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"XGBFlatECClassifier expects pooled 2D inputs, got shape {X_np.shape}")

        y_np, target_mask_np = self._normalize_targets(y, target_mask)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows for XGBFlatECClassifier. "
                f"Received X={X_np.shape[0]}, y={y_np.shape[0]}."
            )

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            if X_val_np.ndim != 2:
                raise ValueError(f"X_val must be 2D, got shape {X_val_np.shape}")
            y_val_np, target_mask_val_np = self._normalize_targets(y_val, target_mask_val)
            if X_val_np.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows for XGBFlatECClassifier. "
                    f"Received X_val={X_val_np.shape[0]}, y_val={y_val_np.shape[0]}."
                )
            val_payload = (X_val_np, y_val_np, target_mask_val_np)

        self.n_outputs_ = y_np.shape[1]
        self.models_ = []
        self.constant_predictions_ = []

        for column_index in range(self.n_outputs_):
            known_mask = target_mask_np[:, column_index]
            if not np.any(known_mask):
                self.models_.append(None)
                self.constant_predictions_.append(np.nan)
                continue

            y_column = y_np[known_mask, column_index].astype(np.int32, copy=False)
            unique_classes = np.unique(y_column)
            if unique_classes.size == 1:
                self.models_.append(None)
                self.constant_predictions_.append(float(unique_classes[0]))
                continue

            X_column = X_np[known_mask]
            use_early_stopping = False
            fit_kwargs = {"verbose": False}
            if val_payload is not None:
                X_val_np, y_val_np, target_mask_val_np = val_payload
                val_known_mask = target_mask_val_np[:, column_index]
                if np.any(val_known_mask):
                    fit_kwargs["eval_set"] = [
                        (
                            X_val_np[val_known_mask],
                            y_val_np[val_known_mask, column_index].astype(np.int32, copy=False),
                        )
                    ]
                    use_early_stopping = True

            model = self._make_model(use_early_stopping=use_early_stopping)
            model.fit(X_column, y_column, **fit_kwargs)
            self.models_.append(model)
            self.constant_predictions_.append(None)

        return self

    def predict(self, X) -> np.ndarray:
        if self.n_outputs_ is None:
            raise ValueError("XGBFlatECClassifier must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")

        pred = np.zeros((X_np.shape[0], self.n_outputs_), dtype=np.float32)
        for column_index, (model, constant_prediction) in enumerate(
            zip(self.models_, self.constant_predictions_)
        ):
            if constant_prediction is not None:
                if np.isnan(constant_prediction):
                    pred[:, column_index] = np.nan
                else:
                    pred[:, column_index] = np.float32(constant_prediction)
                continue

            pred[:, column_index] = model.predict(X_np).astype(np.float32, copy=False)
        return pred


class ResidueBinaryClassifierTorch:
    """Binary classifier over per-residue embeddings with padding-aware BCE loss."""

    def __init__(
        self,
        hidden_dims=(),
        lr=1e-3,
        epochs=200,
        optimizer_name="adam",
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None

    @staticmethod
    def _batch_to_input(batch, device: torch.device):
        X_t = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(device)
        y_t = torch.from_numpy(np.asarray(batch.target_per_residue, dtype=np.float32)).to(device)
        target_mask_t = torch.from_numpy(np.asarray(batch.target_mask, dtype=bool)).to(device)
        return X_t, y_t, target_mask_t

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        if batch_kind != "protein_residue":
            raise ValueError(f"ResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'")

        first_batch = next(iter(dataloader))
        X_t, _, _ = self._batch_to_input(first_batch, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        optimizer = _make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")


        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits, y_t)
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                logits = self.model(X_t).squeeze(-1)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                    logits = self.model(X_t).squeeze(-1)
                    losses.append(float(compute_batch_loss(logits, y_t, target_mask_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_dataloader is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(f"ResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'")

        pred_rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_t, _, target_mask_t = self._batch_to_input(batch, self.device)
                logits = self.model(X_t).squeeze(-1)
                batch_pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
                batch_mask = target_mask_t.cpu().numpy().astype(bool)
                for pred_row, mask_row in zip(batch_pred, batch_mask):
                    pred_rows.append(pred_row[mask_row])
        return _pad_1d_array_list(pred_rows, dtype=np.float32)


class XGBResidueBinaryClassifier:
    """XGBoost binary classifier over residue embeddings.

    Each valid residue becomes one training example by flattening a padded
    batch of shape (batch, seq_len, emb_dim) into (n_valid_residues, emb_dim)
    using the target mask.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        early_stopping_rounds=None,
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use XGBResidueBinaryClassifier.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.constant_prediction_ = None

    @staticmethod
    def _flatten_batch(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(batch.protein_emb, dtype=np.float32)
        y = np.asarray(batch.target_per_residue, dtype=np.float32)
        mask = np.asarray(batch.target_mask, dtype=bool)
        return (
            X[mask].astype(np.float32, copy=False),
            y[mask].astype(np.int32, copy=False),
            mask.sum(axis=1).astype(np.int32, copy=False),
        )

    @classmethod
    def _flatten_loader(cls, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        lengths: list[int] = []

        for batch in dataloader:
            X_flat, y_flat, batch_lengths = cls._flatten_batch(batch)
            if X_flat.size == 0:
                continue
            X_chunks.append(X_flat)
            y_chunks.append(y_flat)
            lengths.extend(batch_lengths.tolist())

        if not X_chunks:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.concatenate(X_chunks, axis=0),
            np.concatenate(y_chunks, axis=0),
            np.asarray(lengths, dtype=np.int32),
        )

    @staticmethod
    def _split_flat_predictions(flat_pred: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        start = 0
        for length in lengths.tolist():
            stop = start + int(length)
            rows.append(np.asarray(flat_pred[start:stop], dtype=np.float32))
            start = stop
        return rows

    def _make_model(self, use_early_stopping: bool) -> XGBClassifier:
        params = {
            "objective": "binary:logistic",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
        }
        if use_early_stopping and self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        return XGBClassifier(**params)

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        if batch_kind != "protein_residue":
            raise ValueError(f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'")

        X_train, y_train, _ = self._flatten_loader(dataloader)
        if X_train.size == 0:
            raise ValueError("No residue-level training samples found for XGBoost.")

        unique_classes = np.unique(y_train)
        if unique_classes.size == 1:
            self.constant_prediction_ = float(unique_classes[0])
            self.model = None
            return self

        X_val = y_val = None
        if val_dataloader is not None:
            X_val, y_val, _ = self._flatten_loader(val_dataloader)

        use_early_stopping = X_val is not None and X_val.size > 0
        self.constant_prediction_ = None
        self.model = self._make_model(use_early_stopping=use_early_stopping)
        fit_kwargs = {"verbose": False}
        if use_early_stopping:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'")

        if self.model is None and self.constant_prediction_ is None:
            raise ValueError("XGBResidueBinaryClassifier must be fit before calling predict_loader().")

        pred_rows: list[np.ndarray] = []
        for batch in dataloader:
            X_flat, _, lengths = self._flatten_batch(batch)
            if self.constant_prediction_ is not None:
                flat_pred = np.full((X_flat.shape[0],), self.constant_prediction_, dtype=np.float32)
            else:
                flat_pred = self.model.predict(X_flat).astype(np.float32, copy=False)
            pred_rows.extend(self._split_flat_predictions(flat_pred, lengths))

        return _pad_1d_array_list(pred_rows, dtype=np.float32)


MODEL_MAP = {
    "classification": {
        "mlp_classification": {
            "wrap": MLPClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "sgd_classification": {
            "wrap": SGDClassifierTorch,
            "params": {
                "lr": 0.01,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    },
    "regression": {
        "mlp_regressor": {
            "wrap": MLPRegressorTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "sgd_regressor": {
            "wrap": SGDRegressorTorch,
            "params": {
                "lr": 0.01,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    }
}

if XGBClassifier is not None:
    MODEL_MAP["classification"]["xgb_classification"] = {
        "wrap": XGBClassifier,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"},
    }

if XGBRegressor is not None:
    MODEL_MAP["regression"]["xgb_regression"] = {
        "wrap": XGBRegressor,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"},
    }


STRUCTURED_MODEL_MAP = {
    "multilabel_ec": {
        "mlp_classification": {
            "wrap": ECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": False,
        },
        "sgd_classification": {
            "wrap": ECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": False,
        },
    },
    "residue_binary_vector": {
        "mlp_classification": {
            "wrap": ResidueBinaryClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "sgd_classification": {
            "wrap": ResidueBinaryClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    },
}

if XGBClassifier is not None:
    STRUCTURED_MODEL_MAP["residue_binary_vector"]["xgb_classification"] = {
        "wrap": XGBResidueBinaryClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        },
        "residue_aware": True,
    }
    STRUCTURED_MODEL_MAP["multilabel_ec"]["xgb_classification"] = {
        "wrap": XGBFlatECClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        },
        "residue_aware": False,
        "flat_ec_labels": True,
    }
