"""Categorical encoder and real-feature scalers (F3).

The encoder assigns one ``<unk>`` slot per column at index 0; unseen
categories at transform time fold into that slot without raising and
trigger one aggregated :data:`~seq_sklearn.logging.Event.DATA_UNSEEN_CATEGORIES`
record per ``transform`` call. Scalers wrap scikit-learn primitives but
persist their fitted statistics as plain Python floats so the save / load
JSON path (A17) serializes directly without pickling sklearn objects.
"""

import logging
from typing import Literal, Protocol

import numpy as np
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from seq_sklearn.errors import NotFittedError
from seq_sklearn.logging import Event, emit

__all__ = [
    "CategoricalEncoder",
    "IdentityScaler",
    "QuantileUniformScaler",
    "RobustScalerWrapper",
    "Scaler",
    "ScalerStrategy",
    "StandardScalerWrapper",
    "make_scaler",
]

logger = logging.getLogger(__name__)

UNK_TOKEN = "<unk>"
UNK_INDEX = 0


def _jsonable(value: object) -> object:
    """Coerce a numpy scalar / array to a JSON primitive (A17).

    The save / load path persists scaler statistics in ``state.json``;
    numpy arrays become nested lists and numpy scalars become Python
    ``int`` / ``float`` so the JSON encoder never sees a numpy type.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


ScalerStrategy = Literal["standard", "robust", "quantile_uniform", "none"]


class CategoricalEncoder:
    """Per-column categorical vocabulary with a shared-per-column unknown slot.

    ``fit`` builds an ordered vocabulary per column with ``<unk>`` pinned
    at index 0. ``transform`` maps known levels to their index and unseen
    levels to ``<unk>`` (index 0) without raising; it emits exactly one
    aggregated structured log record per call when any unseen level was
    seen. ``inverse_transform`` decodes indices back to level strings; the
    ``<unk>`` index decodes to the literal ``"<unk>"``.
    """

    def __init__(self) -> None:
        self.vocabularies_: dict[str, list[str]] = {}
        self._index_: dict[str, dict[str, int]] = {}
        self._fitted = False

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def fit(self, columns: dict[str, np.ndarray]) -> "CategoricalEncoder":
        """Build a per-column vocabulary with ``<unk>`` at index 0.

        ``columns`` maps a column name to its 1-D array of raw level
        values. Levels are stringified and de-duplicated in first-seen
        sorted order for determinism.
        """
        self.vocabularies_ = {}
        self._index_ = {}
        for name, values in columns.items():
            uniques = sorted({str(v) for v in np.asarray(values).tolist()})
            vocab = [UNK_TOKEN, *uniques]
            self.vocabularies_[name] = vocab
            self._index_[name] = {level: i for i, level in enumerate(vocab)}
        self._fitted = True
        return self

    def transform(self, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Encode each column to int64 indices, folding unseen to ``<unk>``.

        Emits one aggregated
        :data:`~seq_sklearn.logging.Event.DATA_UNSEEN_CATEGORIES` record
        per call naming the total count of rows that fell into the
        unknown slot.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise NotFittedError("CategoricalEncoder must be fitted before transform")
        out: dict[str, np.ndarray] = {}
        unseen_total = 0
        for name, values in columns.items():
            index = self._index_[name]
            arr = np.asarray(values).tolist()
            encoded = np.empty(len(arr), dtype=np.int64)
            for i, raw in enumerate(arr):
                key = str(raw)
                if key in index:
                    encoded[i] = index[key]
                else:
                    encoded[i] = UNK_INDEX
                    unseen_total += 1
            out[name] = encoded
        if unseen_total > 0:
            emit(
                logger,
                Event.DATA_UNSEEN_CATEGORIES,
                level=logging.WARNING,
                count=unseen_total,
            )
        return out

    def inverse_transform(self, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Decode int64 index columns back to level strings.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise NotFittedError("CategoricalEncoder must be fitted before inverse_transform")
        out: dict[str, np.ndarray] = {}
        for name, values in columns.items():
            vocab = self.vocabularies_[name]
            idx = np.asarray(values, dtype=np.int64)
            out[name] = np.array([vocab[i] for i in idx], dtype=object)
        return out

    def get_fitted_state(self) -> dict[str, object]:
        """Return the per-column vocabularies as a JSON-primitive dict (A17).

        ``_index_`` is derived from the vocabularies on load, so only the
        ordered vocabulary lists are persisted.
        """
        if not self._fitted:
            raise NotFittedError("CategoricalEncoder must be fitted before get_fitted_state")
        return {"vocabularies": {name: list(vocab) for name, vocab in self.vocabularies_.items()}}

    def set_fitted_state(self, state: dict[str, object]) -> "CategoricalEncoder":
        """Rebuild from :meth:`get_fitted_state` output (vocab + derived index)."""
        vocabs = state["vocabularies"]
        if not isinstance(vocabs, dict):
            raise ValueError("CategoricalEncoder state 'vocabularies' must be a dict")
        self.vocabularies_ = {name: list(vocab) for name, vocab in vocabs.items()}
        self._index_ = {
            name: {level: i for i, level in enumerate(vocab)}
            for name, vocab in self.vocabularies_.items()
        }
        self._fitted = True
        return self


class Scaler(Protocol):
    """Common scaler surface: fit / transform / inverse_transform on 2-D arrays.

    ``get_fitted_state`` / ``set_fitted_state`` are the A17 save / load
    hooks: the fitted statistics round-trip through ``state.json`` as
    JSON primitives so the scaler is reconstructed without re-fitting
    and without pickling the backing sklearn object.
    """

    def fit(self, x: np.ndarray) -> "Scaler": ...

    def transform(self, x: np.ndarray) -> np.ndarray: ...

    def inverse_transform(self, x: np.ndarray) -> np.ndarray: ...

    def get_fitted_state(self) -> dict[str, object]: ...

    def set_fitted_state(self, state: dict[str, object]) -> "Scaler": ...


class _SklearnBackedScaler:
    """Base wrapper that round-trips through a scikit-learn primitive.

    Subclasses build the primitive in :meth:`_make`; this class drives
    fit / transform / inverse_transform on a ``(n_samples, n_features)``
    array. Statistics live on the sklearn object after fit; the save /
    load layer reads them out as plain floats via
    :meth:`get_fitted_state`.
    """

    def __init__(self) -> None:
        self._impl = self._make()
        self._fitted = False

    def _make(self) -> object:
        raise NotImplementedError

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def fit(self, x: np.ndarray) -> "Scaler":
        self._impl.fit(np.asarray(x, dtype=np.float64))  # type: ignore[attr-defined]
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before transform")
        return np.asarray(
            self._impl.transform(np.asarray(x, dtype=np.float64)),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before inverse_transform")
        return np.asarray(
            self._impl.inverse_transform(np.asarray(x, dtype=np.float64)),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

    # sklearn public fitted attributes that fully determine transform /
    # inverse_transform. Restored verbatim on load so the scaler is
    # reconstructed without re-fitting (no data at load time) and
    # without pickling the sklearn object (A17: no pickle in artifacts).
    _FITTED_ATTRS: tuple[str, ...] = ()

    def get_fitted_state(self) -> dict[str, object]:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before get_fitted_state")
        return {
            attr: _jsonable(getattr(self._impl, attr))  # type: ignore[arg-type]
            for attr in self._FITTED_ATTRS
        }

    def set_fitted_state(self, state: dict[str, object]) -> "Scaler":
        impl = self._make()
        for attr in self._FITTED_ATTRS:
            value = state[attr]
            restored = np.asarray(value, dtype=np.float64) if isinstance(value, list) else value
            setattr(impl, attr, restored)
        self._impl = impl
        self._fitted = True
        return self


class StandardScalerWrapper(_SklearnBackedScaler):
    """Zero-mean unit-variance scaling backed by sklearn ``StandardScaler``."""

    _FITTED_ATTRS = ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_")

    def _make(self) -> object:
        return StandardScaler()


class RobustScalerWrapper(_SklearnBackedScaler):
    """Median / IQR scaling backed by sklearn ``RobustScaler``."""

    _FITTED_ATTRS = ("center_", "scale_", "n_features_in_")

    def _make(self) -> object:
        return RobustScaler()


class QuantileUniformScaler(_SklearnBackedScaler):
    """Rank-to-uniform scaling backed by sklearn ``QuantileTransformer``."""

    _FITTED_ATTRS = ("n_quantiles_", "quantiles_", "references_", "n_features_in_")

    def _make(self) -> object:
        return QuantileTransformer(output_distribution="uniform")


class IdentityScaler:
    """No-op scaler for the ``none`` strategy.

    Returns its input unchanged; still requires :meth:`fit` so the
    fitted-state contract is uniform across strategies.
    """

    def __init__(self) -> None:
        self._fitted = False

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def fit(self, x: np.ndarray) -> "Scaler":  # noqa: ARG002
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("IdentityScaler must be fitted before transform")
        return np.asarray(x, dtype=np.float64)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("IdentityScaler must be fitted before inverse_transform")
        return np.asarray(x, dtype=np.float64)

    def get_fitted_state(self) -> dict[str, object]:
        if not self._fitted:
            raise NotFittedError("IdentityScaler must be fitted before get_fitted_state")
        return {}

    def set_fitted_state(self, state: dict[str, object]) -> "Scaler":  # noqa: ARG002
        self._fitted = True
        return self


def make_scaler(strategy: ScalerStrategy) -> Scaler:
    """Return a scaler instance for ``strategy``.

    Raises:
        ValueError: If ``strategy`` is not one of the four F3 strategies.
    """
    if strategy == "standard":
        return StandardScalerWrapper()
    if strategy == "robust":
        return RobustScalerWrapper()
    if strategy == "quantile_uniform":
        return QuantileUniformScaler()
    if strategy == "none":
        return IdentityScaler()
    raise ValueError(f"unknown scaler strategy {strategy!r}")
