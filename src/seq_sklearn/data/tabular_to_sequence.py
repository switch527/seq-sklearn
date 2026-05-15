"""Panel-to-sequence transformer (A5 / F3).

Converts a validated panel DataFrame into the batched tensor dict every
sequence model in the library consumes. One window per valid entity-time
position, left-padded to ``lookback`` with a ``True = padding`` mask.
"""

import hashlib
import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch

from seq_sklearn._validate import check_columns, check_y
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.data.encoders import (
    CategoricalEncoder,
    Scaler,
    make_scaler,
)
from seq_sklearn.errors import ConfigError, DataContractError, NotFittedError
from seq_sklearn.logging import Event, emit

__all__ = ["TabularToSequence"]

logger = logging.getLogger(__name__)

TaskType = Literal["binary", "multiclass", "regression_point", "regression_quantile"]

_CLASSIFICATION_TASKS = ("binary", "multiclass")


def _embedding_dim(cardinality: int) -> int:
    """fastai embedding-size heuristic ``min(50, round(1.6 * card**0.56))``."""
    return min(50, round(1.6 * cardinality**0.56))


class TabularToSequence:
    """Validated panel in, batched tensor dict out (A5 / F3).

    Construct with a frozen :class:`TabularToSequenceConfig` and the
    ``task_type`` (drives the target tensor dtype: ``long`` for the
    classification tasks, ``float`` for regression). Call :meth:`fit`
    then :meth:`transform`; the dict shape is the cross-family input
    contract and is locked here.

    The resolved per-column categorical embedding dimensions are exposed
    on the fitted transformer as ``embedding_dims_`` for downstream model
    construction.

    The emitted ``entity_id`` tensor carries a contiguous integer code
    (sorted by the stringified original id) rather than the raw id, so
    the diagnostic tensor is always a valid ``LongTensor`` regardless of
    the original id dtype.
    """

    def __init__(self, config: TabularToSequenceConfig, task_type: TaskType) -> None:
        self.config = config
        self.task_type: TaskType = task_type
        self.categorical_encoder_: CategoricalEncoder | None = None
        self.real_scaler_: Scaler | None = None
        self.static_real_scaler_: Scaler | None = None
        self.embedding_dims_: dict[str, int] = {}
        self.cardinalities_: dict[str, int] = {}
        self.hashed_columns_: set[str] = set()
        self.feature_schema_fingerprint_: str | None = None
        self._target_map: dict[tuple[object, object], float] = {}
        self._fitted = False

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    @property
    def _cat_cols(self) -> list[str]:
        cfg = self.config
        return [*cfg.static_categorical_cols, *cfg.time_varying_categorical_cols]

    def _check_tz_consistency(self, X: pd.DataFrame) -> None:  # noqa: N803
        """Raise on a runtime-assembled mixed tz-aware / tz-naive time column.

        Runs before the dtype contract check so a mixed-tz object column
        (which pandas materializes as ``object`` dtype) is reported with
        the specific tz-mixing message rather than the generic
        unsupported-dtype message. A missing column is left for
        :func:`seq_sklearn._validate.check_columns` to report.
        """
        if self.config.time_col not in X.columns:
            return
        time = X[self.config.time_col]
        if time.dtype != object:
            return
        tz_flags = {getattr(v, "tzinfo", None) is not None for v in time}
        if len(tz_flags) > 1:
            raise DataContractError(
                f"time_col {self.config.time_col!r} mixes tz-aware and tz-naive "
                "timestamps within one fit"
            )

    def _resolve_static_real_strategy(
        self,
    ) -> Literal["standard", "robust", "quantile_uniform", "none"]:
        cfg = self.config
        if cfg.scaling_static_real == "inherit":
            return cfg.scaling_real
        return cfg.scaling_static_real

    def _build_fingerprint(self, X: pd.DataFrame) -> str:  # noqa: N803
        """Deterministic sha256 over sorted column names + dtypes + config."""
        cols = sorted(X.columns)
        parts = [f"{c}:{X[c].dtype!s}" for c in cols]
        parts.append(repr(self.config.model_dump()))
        parts.append(self.task_type)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def fit(self, X: pd.DataFrame, y: object) -> "TabularToSequence":  # noqa: N803
        """Fit encoders, scalers, and the schema fingerprint (A5 fit steps).

        Drops entities with fewer than ``min_periods`` real rows before
        fitting statistics.

        Raises:
            DataContractError: Missing columns, duplicate ``(id, time)``,
                unsupported time dtype, or runtime tz mixing.
            ConfigError: A categorical column exceeds
                ``max_categorical_cardinality`` and ``hash_high_cardinality``
                is off.
            ValueError: ``y`` is not single-output (delegated to
                :func:`seq_sklearn._validate.check_y`).
        """
        cfg = self.config
        required = [
            *cfg.static_categorical_cols,
            *cfg.static_real_cols,
            *cfg.time_varying_real_cols,
            *cfg.time_varying_categorical_cols,
        ]
        self._check_tz_consistency(X)
        check_columns(X, id_col=cfg.id_col, time_col=cfg.time_col, required=required)
        y_arr = check_y(y)

        if len(y_arr) != len(X):
            raise DataContractError(f"y length {len(y_arr)} does not match X row count {len(X)}")
        self._target_map = {
            (row_id, row_time): float(label)
            for row_id, row_time, label in zip(
                X[cfg.id_col].to_numpy(),
                X[cfg.time_col].to_numpy(),
                y_arr,
                strict=True,
            )
        }

        id_values = X[cfg.id_col].to_numpy()
        unique_ids, id_counts = np.unique(id_values, return_counts=True)
        keep_ids = {
            uid for uid, n in zip(unique_ids, id_counts, strict=True) if n >= cfg.min_periods
        }
        keep_mask = np.array([v in keep_ids for v in id_values], dtype=bool)
        fit_frame = X.loc[keep_mask]

        encoder = CategoricalEncoder()
        cat_payload: dict[str, np.ndarray] = {}
        for col in self._cat_cols:
            raw = fit_frame[col].to_numpy()
            cardinality = int(np.unique(raw).size)
            if cardinality > cfg.max_categorical_cardinality:
                if not cfg.hash_high_cardinality:
                    raise ConfigError(
                        f"categorical column {col!r} cardinality {cardinality} "
                        f"exceeds max_categorical_cardinality "
                        f"{cfg.max_categorical_cardinality}; hash, bin, or set "
                        "hash_high_cardinality=True"
                    )
                self.hashed_columns_.add(col)
                hashed = self._hash_column(raw, cfg.max_categorical_cardinality)
                cat_payload[col] = hashed
                self.cardinalities_[col] = cfg.max_categorical_cardinality
            else:
                cat_payload[col] = raw
                self.cardinalities_[col] = cardinality
        encoder.fit(cat_payload)
        self.categorical_encoder_ = encoder

        override = dict(cfg.categorical_embed_dims)
        for col in self._cat_cols:
            # +1 for the per-column <unk> slot.
            vocab_size = self.cardinalities_[col] + 1
            self.embedding_dims_[col] = override.get(col, _embedding_dim(vocab_size))

        real_scaler = make_scaler(cfg.scaling_real)
        if cfg.time_varying_real_cols:
            real_scaler.fit(fit_frame[list(cfg.time_varying_real_cols)].to_numpy(dtype=float))
        self.real_scaler_ = real_scaler

        static_scaler = make_scaler(self._resolve_static_real_strategy())
        if cfg.static_real_cols:
            static_scaler.fit(fit_frame[list(cfg.static_real_cols)].to_numpy(dtype=float))
        self.static_real_scaler_ = static_scaler

        self.feature_schema_fingerprint_ = self._build_fingerprint(X)
        self._fitted = True
        return self

    @staticmethod
    def _hash_column(values: np.ndarray, buckets: int) -> np.ndarray:
        out = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            h = hashlib.blake2b(str(v).encode(), digest_size=8).digest()
            out[i] = str(int.from_bytes(h, "big") % buckets)
        return out

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("TabularToSequence must be fitted before transform")

    def transform(self, X: pd.DataFrame) -> dict[str, torch.Tensor]:  # noqa: N803
        """Window the panel into the batched tensor dict (A5 transform steps).

        Entities below ``min_periods_predict`` produce a NaN-target window
        with one aggregated warning per call.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            DataContractError: Same input-contract checks as :meth:`fit`.
        """
        self._require_fitted()
        cfg = self.config
        assert self.categorical_encoder_ is not None
        assert self.real_scaler_ is not None
        assert self.static_real_scaler_ is not None
        required = [
            *cfg.static_categorical_cols,
            *cfg.static_real_cols,
            *cfg.time_varying_real_cols,
            *cfg.time_varying_categorical_cols,
        ]
        self._check_tz_consistency(X)
        check_columns(X, id_col=cfg.id_col, time_col=cfg.time_col, required=required)

        ordered = X.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)
        lookback = cfg.lookback
        n_tv_real = len(cfg.time_varying_real_cols)
        n_tv_cat = len(cfg.time_varying_categorical_cols)
        n_static_real = len(cfg.static_real_cols)
        n_static_cat = len(cfg.static_categorical_cols)

        static_cat_rows: list[np.ndarray] = []
        static_real_rows: list[np.ndarray] = []
        tv_real_rows: list[np.ndarray] = []
        tv_cat_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        target_rows: list[float] = []
        entity_rows: list[int] = []
        below_floor = 0

        ordered_ids = sorted({str(v) for v in ordered[cfg.id_col].to_numpy()})
        entity_codes = {raw: code for code, raw in enumerate(ordered_ids)}

        for entity_id, group in ordered.groupby(cfg.id_col, sort=True):
            n_rows = len(group)
            window = group.iloc[-lookback:] if n_rows > lookback else group
            w_len = len(window)
            pad = lookback - w_len

            tv_real = (
                self.real_scaler_.transform(
                    window[list(cfg.time_varying_real_cols)].to_numpy(dtype=float)
                )
                if n_tv_real
                else np.zeros((w_len, 0), dtype=np.float64)
            )
            if cfg.clip_features is not None and n_tv_real:
                clip = cfg.clip_features
                tv_real = np.clip(tv_real, -clip, clip)
            if pad > 0 and n_tv_real:
                tv_real = np.vstack([np.zeros((pad, n_tv_real)), tv_real])
            elif pad > 0:
                tv_real = np.zeros((lookback, 0), dtype=np.float64)

            if n_tv_cat:
                tv_cat_encoded = self.categorical_encoder_.transform(
                    {c: window[c].to_numpy() for c in cfg.time_varying_categorical_cols}
                )
                tv_cat = np.stack(
                    [tv_cat_encoded[c] for c in cfg.time_varying_categorical_cols], axis=1
                )
                if pad > 0:
                    tv_cat = np.vstack([np.zeros((pad, n_tv_cat), dtype=np.int64), tv_cat])
            else:
                tv_cat = np.zeros((lookback, 0), dtype=np.int64)

            mask = np.zeros(lookback, dtype=bool)
            if pad > 0:
                mask[:pad] = True

            last = window.iloc[-1]
            if n_static_cat:
                static_cat_encoded = self.categorical_encoder_.transform(
                    {c: np.array([last[c]]) for c in cfg.static_categorical_cols}
                )
                static_cat = np.array(
                    [static_cat_encoded[c][0] for c in cfg.static_categorical_cols],
                    dtype=np.int64,
                )
            else:
                static_cat = np.zeros(0, dtype=np.int64)

            if n_static_real:
                static_real = self.static_real_scaler_.transform(
                    np.asarray([[last[c] for c in cfg.static_real_cols]], dtype=float)
                )[0]
                if cfg.clip_features is not None:
                    static_real = np.clip(static_real, -cfg.clip_features, cfg.clip_features)
            else:
                static_real = np.zeros(0, dtype=np.float64)

            if n_rows < cfg.min_periods_predict:
                target = float("nan")
                below_floor += 1
            else:
                target = self._aligned_target(group)

            static_cat_rows.append(static_cat)
            static_real_rows.append(static_real)
            tv_real_rows.append(tv_real)
            tv_cat_rows.append(tv_cat)
            mask_rows.append(mask)
            target_rows.append(target)
            entity_rows.append(entity_codes[str(entity_id)])

        if below_floor > 0:
            emit(
                logger,
                Event.DATA_DUPLICATE_FLOOR_BREACH_COUNT,
                level=logging.WARNING,
                count=below_floor,
            )

        target_dtype = torch.long if self.task_type in _CLASSIFICATION_TASKS else torch.float32
        target_np = np.asarray(target_rows, dtype=np.float64)
        if self.task_type in _CLASSIFICATION_TASKS:
            target_tensor = torch.as_tensor(
                np.nan_to_num(target_np, nan=-1).astype(np.int64), dtype=torch.long
            )
        else:
            target_tensor = torch.as_tensor(target_np, dtype=torch.float32)

        batch = len(entity_rows)
        return {
            "static_categorical": torch.as_tensor(
                np.asarray(static_cat_rows, dtype=np.int64).reshape(batch, n_static_cat),
                dtype=torch.long,
            ),
            "static_real": torch.as_tensor(
                np.asarray(static_real_rows, dtype=np.float64).reshape(batch, n_static_real),
                dtype=torch.float32,
            ),
            "time_varying_real": torch.as_tensor(
                np.asarray(tv_real_rows, dtype=np.float64).reshape(batch, lookback, n_tv_real),
                dtype=torch.float32,
            ),
            "time_varying_categorical": torch.as_tensor(
                np.asarray(tv_cat_rows, dtype=np.int64).reshape(batch, lookback, n_tv_cat),
                dtype=torch.long,
            ),
            "padding_mask": torch.as_tensor(
                np.asarray(mask_rows, dtype=bool).reshape(batch, lookback),
                dtype=torch.bool,
            ),
            "target": target_tensor.to(target_dtype),
            "entity_id": torch.as_tensor(np.asarray(entity_rows, dtype=np.int64), dtype=torch.long),
        }

    def fit_transform(self, X: pd.DataFrame, y: object) -> dict[str, torch.Tensor]:  # noqa: N803
        """Fit then transform on the same panel."""
        return self.fit(X, y).transform(X)

    def _aligned_target(self, group: pd.DataFrame) -> float:
        """Target ``prediction_step`` rows after the window end, capped at horizon.

        Looks the label up in the fit-time ``(id, time)`` target map.
        Returns NaN when the prediction-step row has no mapped label
        (predict-time future rows, or rows outside the fitted panel).
        """
        cfg = self.config
        n_rows = len(group)
        window_end = n_rows - 1
        target_pos = min(window_end + cfg.prediction_step, n_rows - 1)
        target_row = group.iloc[target_pos]
        key = (target_row[cfg.id_col], target_row[cfg.time_col])
        return self._target_map.get(key, float("nan"))

    def inverse_transform(self, batch: dict[str, torch.Tensor]) -> pd.DataFrame:
        """Reverse scaling and decode categoricals back to a flat frame.

        Operates on the window-end (most recent) timestep of each
        sequence so the result is one row per window.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            NotImplementedError: If any categorical column was hash-tricked
                (the hash is not invertible).
        """
        self._require_fitted()
        cfg = self.config
        assert self.categorical_encoder_ is not None
        assert self.real_scaler_ is not None
        assert self.static_real_scaler_ is not None
        if self.hashed_columns_:
            raise NotImplementedError(
                f"inverse_transform cannot decode hash-tricked columns: "
                f"{sorted(self.hashed_columns_)}"
            )

        frame: dict[str, np.ndarray] = {}
        if cfg.static_categorical_cols:
            static_cat = batch["static_categorical"].cpu().numpy()
            decoded = self.categorical_encoder_.inverse_transform(
                {c: static_cat[:, i] for i, c in enumerate(cfg.static_categorical_cols)}
            )
            for c in cfg.static_categorical_cols:
                frame[c] = decoded[c]

        if cfg.static_real_cols:
            static_real = batch["static_real"].cpu().numpy()
            unscaled = self.static_real_scaler_.inverse_transform(static_real)
            for i, c in enumerate(cfg.static_real_cols):
                frame[c] = unscaled[:, i]

        if cfg.time_varying_real_cols:
            tv_real = batch["time_varying_real"].cpu().numpy()[:, -1, :]
            unscaled = self.real_scaler_.inverse_transform(tv_real)
            for i, c in enumerate(cfg.time_varying_real_cols):
                frame[c] = unscaled[:, i]

        if cfg.time_varying_categorical_cols:
            tv_cat = batch["time_varying_categorical"].cpu().numpy()[:, -1, :]
            decoded = self.categorical_encoder_.inverse_transform(
                {c: tv_cat[:, i] for i, c in enumerate(cfg.time_varying_categorical_cols)}
            )
            for c in cfg.time_varying_categorical_cols:
                frame[c] = decoded[c]

        frame[cfg.id_col] = batch["entity_id"].cpu().numpy()
        return pd.DataFrame(frame)
