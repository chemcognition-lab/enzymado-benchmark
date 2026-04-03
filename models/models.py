from xgboost import XGBRegressor, XGBClassifier
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


def _masked_mean_pool(out: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool (batch, seq_len, dim) → (batch, dim) using boolean mask."""
    if mask is None:
        return out.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()           # (batch, seq_len, 1)
    return (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SGDRegressorTorch:
    """SGDModel wrapper for regression. Accepts 2D (n, emb) or 3D (n, seq, emb) + mask."""

    def __init__(self, lr=0.01, epochs=200, device=None):
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if device else DEVICE
        self.model = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)                      # (..., output_dim)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)   # (batch, output_dim)
        return out

    def fit(self, X, y, mask=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = SGDModel(input_dim, output_dim=1).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str):
        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = SGDModel(input_dim, output_dim=1).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()
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

    def __init__(self, lr=0.01, epochs=200, device=None):
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.classes_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None):
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
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()
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

    def fit_loader(self, dataloader, batch_kind: str):
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
        self.model.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()
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

    def __init__(self, hidden_dims=(512, 256), lr=1e-3, epochs=200, device=None):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if device else DEVICE
        self.model = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None):
        X_t, mask_t, input_dim = _prepare_inputs(X, mask, self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1).to(self.device)
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X, mask=None):
        X_t, mask_t, _ = _prepare_inputs(X, mask, self.device)
        self.model.eval()
        with torch.no_grad():
            return self._forward(X_t, mask_t).squeeze(1).cpu().numpy()

    def fit_loader(self, dataloader, batch_kind: str):
        first_batch = next(iter(dataloader))
        X_t, _, _ = _batch_to_input(first_batch, batch_kind, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()
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

    def __init__(self, hidden_dims=(512, 256), lr=1e-3, epochs=200, device=None):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.classes_ = None

    def _forward(self, X_t: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        out = self.model(X_t)
        if X_t.ndim == 3:
            out = _masked_mean_pool(out, mask_t)
        return out

    def fit(self, X, y, mask=None):
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
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._forward(X_t, mask_t), y_t)
            loss.backward()
            optimizer.step()
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

    def fit_loader(self, dataloader, batch_kind: str):
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
        self.model.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                X_t, mask_t, y_np = _batch_to_input(batch, batch_kind, self.device)
                if n_classes == 2:
                    y_t = torch.from_numpy(y_np).unsqueeze(1).to(self.device)
                else:
                    y_t = torch.tensor([label_map[v] for v in y_np], dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                criterion(self._forward(X_t, mask_t), y_t).backward()
                optimizer.step()
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


MODEL_MAP = {
    "classification": {
        "xgb_classification": {"wrap": XGBClassifier, "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"}},
        "mlp_classification": {"wrap": MLPClassifierTorch, "params": {"hidden_dims": (512, 256), "lr": 1e-3, "epochs": 200}, "residue_aware": True},
        "sgd_classification": {"wrap": SGDClassifierTorch, "params": {"lr": 0.01, "epochs": 200}, "residue_aware": True},
    },
    "regression": {
        "xgb_regression": {"wrap": XGBRegressor, "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"}},
        "mlp_regressor": {"wrap": MLPRegressorTorch, "params": {"hidden_dims": (512, 256), "lr": 1e-3, "epochs": 200}, "residue_aware": True},
        "sgd_regressor": {"wrap": SGDRegressorTorch, "params": {"lr": 0.01, "epochs": 200}, "residue_aware": True},
    }
}

