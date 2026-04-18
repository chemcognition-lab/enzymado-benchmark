"""XGBoost wrappers for standard, residue-level, and flat EC tasks."""

from __future__ import annotations

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError:
    xgb = None
    XGBClassifier = None
    XGBRegressor = None

import numpy as np
import torch

from models.common import pad_1d_array_list


class XGBMultiOutputFlatECExperimental:
    """Experimental multi-output XGBoost baseline for dense flat EC labels.

    This wrapper is intended for benchmarking against the current one-model-per-column
    implementation. It requires a dense multi-hot label matrix with no unknown entries,
    so callers should filter rows with incomplete labels before using it.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.1,
        max_bin=128,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        tree_method="hist",
        multi_strategy="multi_output_tree",
    ):
        if xgb is None:
            raise ImportError("xgboost is required to use XGBMultiOutputFlatECExperimental.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.multi_strategy = multi_strategy
        self.booster = None
        self.n_outputs_: int | None = None

    @staticmethod
    def _prepare_dense_targets(y, target_mask=None) -> np.ndarray:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(
                "XGBMultiOutputFlatECExperimental expects 2D dense multi-output targets, "
                f"got shape {y_np.shape}."
            )

        if target_mask is None:
            return y_np

        mask_np = np.asarray(target_mask, dtype=bool)
        if mask_np.shape != y_np.shape:
            raise ValueError(
                "target_mask shape must match y shape for XGBMultiOutputFlatECExperimental. "
                f"Received mask={mask_np.shape}, y={y_np.shape}."
            )
        if not np.all(mask_np):
            raise ValueError(
                "XGBMultiOutputFlatECExperimental requires dense labels with no unknown entries. "
                "Filter rows/columns with incomplete EC labels before fitting."
            )
        return y_np

    def fit(self, X, y, target_mask=None, X_val=None, y_val=None, target_mask_val=None):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(
                "XGBMultiOutputFlatECExperimental expects pooled 2D inputs, "
                f"got shape {X_np.shape}."
            )
        y_np = self._prepare_dense_targets(y, target_mask)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows for "
                "XGBMultiOutputFlatECExperimental. "
                f"Received X={X_np.shape[0]}, y={y_np.shape[0]}."
            )

        dtrain = xgb.QuantileDMatrix(X_np, label=y_np)
        params = {
            "objective": "binary:logistic",
            "tree_method": self.tree_method,
            "device": self.device,
            "multi_strategy": self.multi_strategy,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "nthread": self.n_jobs,
            "seed": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }

        evals = []
        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            y_val_np = self._prepare_dense_targets(y_val, target_mask_val)
            if X_val_np.ndim != 2:
                raise ValueError(f"X_val must be 2D, got shape {X_val_np.shape}")
            if X_val_np.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows for "
                    "XGBMultiOutputFlatECExperimental. "
                    f"Received X_val={X_val_np.shape[0]}, y_val={y_val_np.shape[0]}."
                )
            dval = xgb.QuantileDMatrix(X_val_np, label=y_val_np, ref=dtrain)
            evals.append((dval, "validation"))

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            verbose_eval=False,
        )
        self.n_outputs_ = y_np.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        if self.booster is None or self.n_outputs_ is None:
            raise ValueError("XGBMultiOutputFlatECExperimental must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        dtest = xgb.QuantileDMatrix(X_np)
        pred = self.booster.predict(dtest)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(X_np.shape[0], self.n_outputs_)
        return (pred >= 0.5).astype(np.float32, copy=False)


class XGBFlatECClassifier:
    """One binary XGBoost model per concatenated EC column."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        max_bin=256,
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
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.models_: list[XGBClassifier | None] = []
        self.constant_predictions_: list[float | None] = []
        self.n_outputs_: int | None = None

    @staticmethod
    def _normalize_targets(y, target_mask=None) -> tuple[np.ndarray, np.ndarray]:
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
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "max_bin": self.max_bin,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
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

            booster = model.get_booster()
            try:
                raw_pred = booster.inplace_predict(X_np, validate_features=False)
                raw_pred = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
                pred[:, column_index] = (raw_pred >= 0.5).astype(np.float32, copy=False)
            except Exception:
                pred[:, column_index] = model.predict(X_np).astype(np.float32, copy=False)
        return pred


class XGBResidueBinaryClassifier:
    """XGBoost binary classifier over residue embeddings."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        max_bin=256,
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
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.max_bin = max_bin
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

        if hasattr(batch, "lengths"):
            lengths = np.asarray(batch.lengths, dtype=np.int32)
            if X.ndim != 2:
                raise ValueError(
                    "Concatenated residue batches must expose a 2D protein_emb array. "
                    f"Received shape {X.shape}."
                )
            if y.ndim != 1:
                raise ValueError(
                    "Concatenated residue batches must expose a 1D target_per_residue array. "
                    f"Received shape {y.shape}."
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "Concatenated residue batch protein_emb and target_per_residue length mismatch: "
                    f"X={X.shape[0]}, y={y.shape[0]}."
                )
            if lengths.sum() != X.shape[0]:
                raise ValueError(
                    "Concatenated residue batch lengths must sum to the flattened residue count. "
                    f"sum(lengths)={lengths.sum()}, X={X.shape[0]}."
                )
            return X.astype(np.float32, copy=False), y.astype(np.int32, copy=False), lengths

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
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "max_bin": self.max_bin,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
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
                booster = self.model.get_booster()
                try:
                    raw_pred = booster.inplace_predict(X_flat, validate_features=False)
                    flat_pred = (np.asarray(raw_pred, dtype=np.float32).reshape(-1) >= 0.5).astype(
                        np.float32,
                        copy=False,
                    )
                except Exception:
                    flat_pred = self.model.predict(X_flat).astype(np.float32, copy=False)
            pred_rows.extend(self._split_flat_predictions(flat_pred, lengths))

        return pad_1d_array_list(pred_rows, dtype=np.float32)
