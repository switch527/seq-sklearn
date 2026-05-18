"""Panel-to-sequence transformer (A5 / F3).

Converts a validated panel DataFrame into the batched tensor dict every
sequence model in the library consumes. Per entity, one sliding window
per window-end position (``n`` windows for an ``n``-row entity), each
spanning the most recent ``lookback`` rows ending at that position,
left-padded to ``lookback`` with a ``True = padding`` mask. Windows for
one entity are contiguous and time-ascending in the output batch so the
Trainer can reconstruct ``window_time_index`` as ``np.arange(n)``.
"""

import hashlib
import logging
from typing import Literal, cast

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

# Private positional-index column attached before the (id, time) sort so
# the caller's original X row order can be restored at predict time. The
# name is namespaced to avoid colliding with a user column; transform()
# raises DataContractError if a caller X already carries it.
_POS = "__seq_sklearn_input_pos__"


def _embedding_dim(cardinality: int) -> int:
    """fastai embedding-size heuristic ``min(50, round(1.6 * card**0.56))``."""
    return min(50, round(1.6 * cardinality**0.56))


def _restore_permutation(emitted_pos: list[int] | np.ndarray, n: int) -> np.ndarray:
    """Inverse permutation that restores caller ``X`` row order.

    ``emitted_pos[k]`` is the caller-``X`` positional index of the
    ``k``-th emitted window. ``transform()`` emits exactly one window
    per input row and drops none, so ``emitted_pos`` is a dense
    permutation of ``0..n-1``; ``np.argsort`` of a dense permutation is
    its inverse, so indexing any emission-order array by the result
    restores caller order. The length check is the sole safety net for
    that precondition (factored here so it is exercised by a direct
    unit call, not only an unconstructable data path).

    Raises:
        DataContractError: ``len(emitted_pos) != n`` (the
            one-window-per-row precondition was violated).
    """
    arr = np.asarray(emitted_pos)
    if len(arr) != n:
        raise DataContractError(
            f"row-order restore precondition violated: emitted {len(arr)} "
            f"windows for {n} input rows (expected exactly one per row)"
        )
    return np.argsort(arr, kind="stable")


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
    assigned in batch-emission order (code ``k`` is the ``k``-th entity
    block in the output) rather than the raw id, so the diagnostic tensor
    is always a valid ``LongTensor`` regardless of the original id dtype
    and is monotone non-decreasing across the batch.
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

    def serialize(self) -> dict[str, object]:
        """Return the fitted transformer state as a JSON-primitive dict (A17).

        Captures exactly what :meth:`transform` consumes: the categorical
        encoder vocabularies, the two scalers' fitted statistics, the
        resolved embedding dims / cardinalities, the hashed-column set,
        and the schema fingerprint. The fit-time ``(id, time)`` target
        map is deliberately NOT persisted: it is a training artifact
        (:meth:`_aligned_target` falls back to NaN for unmapped rows, so
        a reloaded estimator predicting on fresh data is unaffected) and
        a model artifact must not carry training labels.

        A scaler is serialized only when its feature group is present
        (the exact gate :meth:`fit` uses to call ``scaler.fit``); an
        absent group leaves the scaler unfitted on both fit and reload.
        """
        self._require_fitted()
        cfg = self.config
        assert self.categorical_encoder_ is not None
        assert self.real_scaler_ is not None
        assert self.static_real_scaler_ is not None
        return {
            "categorical_encoder": self.categorical_encoder_.get_fitted_state(),
            "real_scaler": (
                self.real_scaler_.get_fitted_state() if cfg.time_varying_real_cols else None
            ),
            "static_real_scaler": (
                self.static_real_scaler_.get_fitted_state() if cfg.static_real_cols else None
            ),
            "embedding_dims": dict(self.embedding_dims_),
            "cardinalities": dict(self.cardinalities_),
            "hashed_columns": sorted(self.hashed_columns_),
            "feature_schema_fingerprint": self.feature_schema_fingerprint_,
        }

    @classmethod
    def deserialize(
        cls,
        blob: dict[str, object],
        config: TabularToSequenceConfig,
        task_type: TaskType,
    ) -> "TabularToSequence":
        """Rebuild a fitted transformer from :meth:`serialize` output.

        ``config`` / ``task_type`` are owned by the estimator's
        ``state.json`` (not the transformer blob) and passed back in.
        Scalers are reconstructed via :func:`make_scaler` with the same
        strategy ``fit`` used, then restored from fitted state when their
        feature group was present.
        """
        obj = cls(config, task_type)
        encoder = CategoricalEncoder()
        encoder.set_fitted_state(cast(dict[str, object], blob["categorical_encoder"]))
        obj.categorical_encoder_ = encoder

        real_scaler = make_scaler(config.scaling_real)
        real_state = blob["real_scaler"]
        if real_state is not None:
            real_scaler.set_fitted_state(cast(dict[str, object], real_state))
        obj.real_scaler_ = real_scaler

        static_scaler = make_scaler(obj._resolve_static_real_strategy())
        static_state = blob["static_real_scaler"]
        if static_state is not None:
            static_scaler.set_fitted_state(cast(dict[str, object], static_state))
        obj.static_real_scaler_ = static_scaler

        obj.embedding_dims_ = dict(cast(dict[str, int], blob["embedding_dims"]))
        obj.cardinalities_ = dict(cast(dict[str, int], blob["cardinalities"]))
        obj.hashed_columns_ = set(cast(list[str], blob["hashed_columns"]))
        fingerprint = blob["feature_schema_fingerprint"]
        obj.feature_schema_fingerprint_ = None if fingerprint is None else str(fingerprint)
        obj._fitted = True
        return obj

    def transform(self, X: pd.DataFrame) -> dict[str, torch.Tensor]:  # noqa: N803
        """Window the panel into the batched tensor dict (A5 transform steps).

        Per entity (rows sorted by time, positions ``0..n-1``), emits one
        window per window-end position ``e``: the window spans rows
        ``[max(0, e - lookback + 1) .. e]``, left-padded to ``lookback``,
        with target at ``min(e + prediction_step, n - 1)`` and
        ``window_time_index`` value ``e``. Windows for one entity are
        contiguous and time-ascending in the output batch. The emitted
        ``entity_id`` code is assigned in batch-emission order: code
        ``k`` is the ``k``-th entity block, so ``entity_id`` is monotone
        non-decreasing across the batch.

        The ``min_periods_predict`` gate is per entity on the entity's
        real row count: an entity below the floor still yields all of its
        NaN-target windows and counts once toward the single aggregated
        warning emitted per call.

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

        if _POS in X.columns:
            raise DataContractError(
                f"input column name {_POS!r} collides with the private "
                "row-order restore key; rename the offending column"
            )
        # Attach the original caller-X positional index BEFORE the sort so
        # it survives into emission order; reading it back per window lets
        # predict restore the caller's input X row order (F1).
        ordered = (
            X.assign(**{_POS: np.arange(len(X))})
            .sort_values([cfg.id_col, cfg.time_col], kind="stable")
            .reset_index(drop=True)
        )
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
        emitted_pos: list[int] = []
        below_floor = 0

        for entity_code, (_, group) in enumerate(ordered.groupby(cfg.id_col, sort=True)):
            n_rows = len(group)
            below = n_rows < cfg.min_periods_predict
            if below:
                below_floor += 1
            # _POS values for this group, in the sorted (id, time) order
            # the window loop iterates; group-local window_end indexes it.
            group_pos = group[_POS].to_numpy()

            entity_tv_real = (
                self.real_scaler_.transform(
                    group[list(cfg.time_varying_real_cols)].to_numpy(dtype=float)
                )
                if n_tv_real
                else np.zeros((n_rows, 0), dtype=np.float64)
            )
            if cfg.clip_features is not None and n_tv_real:
                clip = cfg.clip_features
                entity_tv_real = np.clip(entity_tv_real, -clip, clip)

            if n_tv_cat:
                tv_cat_encoded = self.categorical_encoder_.transform(
                    {c: group[c].to_numpy() for c in cfg.time_varying_categorical_cols}
                )
                entity_tv_cat = np.stack(
                    [tv_cat_encoded[c] for c in cfg.time_varying_categorical_cols], axis=1
                )
            else:
                entity_tv_cat = np.zeros((n_rows, 0), dtype=np.int64)

            # Static features are entity-constant (F2), so encode/scale
            # them once per entity rather than once per window. Encoding
            # inside the window loop would also re-fire the aggregated
            # unseen-category log once per window for one static value.
            first = group.iloc[0]
            if n_static_cat:
                static_cat_encoded = self.categorical_encoder_.transform(
                    {c: np.array([first[c]]) for c in cfg.static_categorical_cols}
                )
                static_cat = np.array(
                    [static_cat_encoded[c][0] for c in cfg.static_categorical_cols],
                    dtype=np.int64,
                )
            else:
                static_cat = np.zeros(0, dtype=np.int64)

            if n_static_real:
                static_real = self.static_real_scaler_.transform(
                    np.asarray([[first[c] for c in cfg.static_real_cols]], dtype=float)
                )[0]
                if cfg.clip_features is not None:
                    static_real = np.clip(static_real, -cfg.clip_features, cfg.clip_features)
            else:
                static_real = np.zeros(0, dtype=np.float64)

            for window_end in range(n_rows):
                window_start = max(0, window_end - lookback + 1)
                w_len = window_end - window_start + 1
                pad = lookback - w_len

                tv_real_window = entity_tv_real[window_start : window_end + 1]
                if pad > 0 and n_tv_real:
                    tv_real_window = np.vstack([np.zeros((pad, n_tv_real)), tv_real_window])
                elif pad > 0:
                    tv_real_window = np.zeros((lookback, 0), dtype=np.float64)

                tv_cat_window = entity_tv_cat[window_start : window_end + 1]
                if pad > 0:
                    tv_cat_window = np.vstack(
                        [np.zeros((pad, n_tv_cat), dtype=np.int64), tv_cat_window]
                    )

                mask = np.zeros(lookback, dtype=bool)
                if pad > 0:
                    mask[:pad] = True

                target = float("nan") if below else self._aligned_target(group, window_end)

                static_cat_rows.append(static_cat)
                static_real_rows.append(static_real)
                tv_real_rows.append(tv_real_window)
                tv_cat_rows.append(tv_cat_window)
                mask_rows.append(mask)
                target_rows.append(target)
                entity_rows.append(entity_code)
                emitted_pos.append(int(group_pos[window_end]))

        if below_floor > 0:
            emit(
                logger,
                Event.DATA_DUPLICATE_FLOOR_BREACH_COUNT,
                level=logging.WARNING,
                count=below_floor,
                min_periods_predict=cfg.min_periods_predict,
            )

        target_dtype = torch.long if self.task_type in _CLASSIFICATION_TASKS else torch.float32
        target_np = np.asarray(target_rows, dtype=np.float64)
        if self.task_type in _CLASSIFICATION_TASKS:
            target_tensor = torch.as_tensor(
                np.nan_to_num(target_np, nan=-1).astype(np.int64), dtype=torch.long
            )
        else:
            target_tensor = torch.as_tensor(target_np, dtype=torch.float32)

        input_row_order = _restore_permutation(emitted_pos, len(X))

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
            # Per-call, stateless (I-B): indexing any emission-order array
            # by this restores the caller's input X row order (F1).
            "input_row_order": torch.as_tensor(
                np.asarray(input_row_order, dtype=np.int64), dtype=torch.long
            ),
        }

    def fit_transform(self, X: pd.DataFrame, y: object) -> dict[str, torch.Tensor]:  # noqa: N803
        """Encoder and scaler statistics are drawn from ``X`` itself."""
        return self.fit(X, y).transform(X)

    def _aligned_target(self, group: pd.DataFrame, window_end: int) -> float:
        """Target ``prediction_step`` rows after ``window_end``, capped at horizon.

        ``window_end`` is the 0-based position of the window's last real
        row within the entity. The target row is
        ``min(window_end + prediction_step, n_rows - 1)`` (the cap keeps
        the window instead of dropping it). Looks the label up in the
        fit-time ``(id, time)`` target map; returns NaN when the
        prediction-step row has no mapped label (predict-time future
        rows, or rows outside the fitted panel).
        """
        cfg = self.config
        n_rows = len(group)
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
