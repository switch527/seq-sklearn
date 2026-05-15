"""BaseEstimator adapter for the frozen :class:`TabularToSequenceConfig`.

Per architecture A4 step 3, sklearn's mutation contract on nested
estimators (``set_params(tabular_config__lookback=6)``) is reconciled
with pydantic's ``frozen=True`` by wrapping the pydantic config in a
thin :class:`sklearn.base.BaseEstimator` adapter whose fields mirror the
pydantic schema 1:1. The outer estimator stores the adapter instance;
``get_params(deep=True)`` recurses automatically.
"""

from collections.abc import Mapping
from typing import Literal

from sklearn.base import BaseEstimator

from seq_sklearn.config.tabular import TabularToSequenceConfig

__all__ = ["TabularConfigParams"]


class TabularConfigParams(BaseEstimator):
    """Mutable mirror of :class:`TabularToSequenceConfig`.

    The field NAMES and INPUT TYPES match the pydantic schema 1:1 so
    callers can pass a dict at construction time and get back a frozen
    pydantic config via :meth:`to_pydantic`. The STORAGE form may
    differ: for instance, ``categorical_embed_dims`` accepts a
    ``Mapping[str, int]`` here for ergonomic input but is normalized to
    a sorted ``tuple[tuple[str, int], ...]`` inside the pydantic model
    so the frozen instance is hashable. Adapter callers see the input
    type; pydantic-side consumers see the canonical storage type.

    :meth:`to_pydantic` is the only sanctioned bridge between the
    mutable sklearn surface and the immutable pydantic surface.

    sklearn-side semantics:

    * ``get_params(deep=False)`` returns the mutable scalar dict; the
      outer estimator's ``get_params(deep=True)`` produces flat
      double-underscore keys (``tabular_config__lookback``).
    * ``set_params(lookback=6)`` mutates in place; the outer
      estimator's ``set_params(tabular_config__lookback=6)`` chains via
      sklearn's standard traversal.
    * ``sklearn.base.clone(adapter)`` produces an independent instance
      (the outer estimator's ``__init__`` calls ``clone`` to defend
      against aliasing).
    """

    # Class-level annotations carry the Literal narrowing through instance
    # attribute access; without these, pyright widens the type to plain
    # `str` after assignment in __init__ and to_pydantic() then fails type
    # checking when passing the value to the pydantic constructor.
    id_col: str
    time_col: str
    static_categorical_cols: tuple[str, ...]
    static_real_cols: tuple[str, ...]
    time_varying_real_cols: tuple[str, ...]
    time_varying_categorical_cols: tuple[str, ...]
    lookback: int
    prediction_step: int
    min_periods: int
    min_periods_predict: int
    scaling_real: Literal["standard", "robust", "quantile_uniform", "none"]
    scaling_static_real: Literal["standard", "robust", "quantile_uniform", "none", "inherit"]
    clip_features: float | None
    max_categorical_cardinality: int
    hash_high_cardinality: bool
    categorical_embed_dims: Mapping[str, int] | None

    def __init__(
        self,
        id_col: str = "id",
        time_col: str = "time",
        static_categorical_cols: tuple[str, ...] = (),
        static_real_cols: tuple[str, ...] = (),
        time_varying_real_cols: tuple[str, ...] = (),
        time_varying_categorical_cols: tuple[str, ...] = (),
        lookback: int = 12,
        prediction_step: int = 1,
        min_periods: int = 1,
        min_periods_predict: int = 1,
        scaling_real: Literal["standard", "robust", "quantile_uniform", "none"] = "standard",
        scaling_static_real: Literal[
            "standard", "robust", "quantile_uniform", "none", "inherit"
        ] = "inherit",
        clip_features: float | None = None,
        max_categorical_cardinality: int = 1000,
        hash_high_cardinality: bool = False,
        categorical_embed_dims: Mapping[str, int] | None = None,
    ) -> None:
        self.id_col = id_col
        self.time_col = time_col
        self.static_categorical_cols = static_categorical_cols
        self.static_real_cols = static_real_cols
        self.time_varying_real_cols = time_varying_real_cols
        self.time_varying_categorical_cols = time_varying_categorical_cols
        self.lookback = lookback
        self.prediction_step = prediction_step
        self.min_periods = min_periods
        self.min_periods_predict = min_periods_predict
        self.scaling_real = scaling_real
        self.scaling_static_real = scaling_static_real
        self.clip_features = clip_features
        self.max_categorical_cardinality = max_categorical_cardinality
        self.hash_high_cardinality = hash_high_cardinality
        self.categorical_embed_dims = categorical_embed_dims

    def to_pydantic(self) -> TabularToSequenceConfig:
        """Build the frozen pydantic config from the current mutable state.

        Pydantic validation errors propagate as :class:`ValidationError`;
        callers (the outer estimator's ``_build_config``) wrap them into
        :class:`seq_sklearn.errors.ConfigError`.
        """
        return TabularToSequenceConfig(
            id_col=self.id_col,
            time_col=self.time_col,
            static_categorical_cols=tuple(self.static_categorical_cols),
            static_real_cols=tuple(self.static_real_cols),
            time_varying_real_cols=tuple(self.time_varying_real_cols),
            time_varying_categorical_cols=tuple(self.time_varying_categorical_cols),
            lookback=self.lookback,
            prediction_step=self.prediction_step,
            min_periods=self.min_periods,
            min_periods_predict=self.min_periods_predict,
            scaling_real=self.scaling_real,
            scaling_static_real=self.scaling_static_real,
            clip_features=self.clip_features,
            max_categorical_cardinality=self.max_categorical_cardinality,
            hash_high_cardinality=self.hash_high_cardinality,
            # The BeforeValidator on TabularToSequenceConfig.categorical_embed_dims
            # accepts None / Mapping / tuple-of-tuples and normalizes to a
            # sorted hashable tuple; no caller-side coercion needed.
            categorical_embed_dims=self.categorical_embed_dims,  # type: ignore[arg-type]
        )
