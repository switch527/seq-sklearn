"""BaseEstimator adapters for the frozen pydantic config family (A4 step 3).

sklearn's mutation contract on nested estimators
(``set_params(tabular_config__lookback=6)``) is reconciled with
pydantic's ``frozen=True`` by wrapping each pydantic config in a thin
:class:`sklearn.base.BaseEstimator` adapter whose fields mirror the
pydantic schema 1:1. The outer estimator stores the adapter instance;
``get_params(deep=True)`` recurses automatically. :meth:`to_pydantic`
is the only sanctioned bridge from the mutable sklearn surface to the
immutable pydantic surface.

Every adapter ``__init__`` is keyword-only (the ``*`` marker): a future
BETA promotion that inserts a typed field cannot shift positional
argument order for existing callers.
"""

from collections.abc import Mapping
from typing import Literal

from sklearn.base import BaseEstimator

from seq_sklearn.config._extras import ExtraValue
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTAdvancedConfig

__all__ = [
    "LossParams",
    "OptimizerParams",
    "SamplerParams",
    "SchedulerParams",
    "TFTAdvancedParams",
    "TabularConfigParams",
]

# Ergonomic input form for an `extra` escape hatch on the mutable
# adapter surface. The pydantic-side ExtraDict BeforeValidator accepts
# None / Mapping / iterable-of-pairs and normalizes to a sorted hashable
# tuple, so the adapter stores the caller's input as-is and to_pydantic
# passes it through unchanged.
_ExtraInput = Mapping[str, ExtraValue] | None


def _as_str_tuple(v: object) -> tuple[str, ...]:
    """Coerce a column-list input to a `tuple[str, ...]`.

    Plain ``tuple(v)`` silently iterates a bare ``str`` character by
    character (``tuple("store_id")`` -> ``('s', 't', 'o', ...)``) which
    pydantic's ``tuple[str, ...]`` validation then accepts as a
    well-typed tuple, and the failure surfaces much later as an opaque
    missing-column error at fit time. Reject the bare-string form
    explicitly with a typed error at construction time.
    """
    if isinstance(v, str):
        raise TypeError(
            "expected a tuple of column names, got a bare str "
            f"{v!r}; wrap it in a tuple: ({v!r},)"
        )
    return tuple(v)  # type: ignore[arg-type]


class TabularConfigParams(BaseEstimator):
    """Mutable mirror of :class:`TabularToSequenceConfig`.

    The field NAMES and INPUT TYPES match the pydantic schema 1:1 so
    callers can pass a dict at construction time and get back a frozen
    pydantic config via :meth:`to_pydantic`. The STORAGE form may
    differ: ``categorical_embed_dims`` accepts a ``Mapping[str, int]``
    here for ergonomic input but is normalized to a sorted
    ``tuple[tuple[str, int], ...]`` inside the pydantic model so the
    frozen instance is hashable.

    sklearn-side semantics:

    * ``get_params(deep=False)`` returns the mutable scalar dict.
    * ``set_params(lookback=6)`` mutates in place.
    * ``sklearn.base.clone(adapter)`` produces an independent instance.
    """

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
        *,
        id_col: str = "id",
        time_col: str = "time",
        static_categorical_cols: tuple[str, ...] = (),
        static_real_cols: tuple[str, ...] = (),
        time_varying_real_cols: tuple[str, ...] = (),
        time_varying_categorical_cols: tuple[str, ...] = (),
        lookback: int = 12,
        prediction_step: int = 0,
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

        Raises:
            pydantic.ValidationError: propagated; the outer estimator's
                ``_build_config`` wraps it into
                :class:`seq_sklearn.errors.ConfigError`.
        """
        return TabularToSequenceConfig(
            id_col=self.id_col,
            time_col=self.time_col,
            static_categorical_cols=_as_str_tuple(self.static_categorical_cols),
            static_real_cols=_as_str_tuple(self.static_real_cols),
            time_varying_real_cols=_as_str_tuple(self.time_varying_real_cols),
            time_varying_categorical_cols=_as_str_tuple(self.time_varying_categorical_cols),
            lookback=self.lookback,
            prediction_step=self.prediction_step,
            min_periods=self.min_periods,
            min_periods_predict=self.min_periods_predict,
            scaling_real=self.scaling_real,
            scaling_static_real=self.scaling_static_real,
            clip_features=self.clip_features,
            max_categorical_cardinality=self.max_categorical_cardinality,
            hash_high_cardinality=self.hash_high_cardinality,
            categorical_embed_dims=self.categorical_embed_dims,  # type: ignore[arg-type]
        )


class OptimizerParams(BaseEstimator):
    """Mutable mirror of :class:`OptimizerConfig`."""

    name: Literal["adamw", "adam", "sgd"]
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    momentum: float
    nesterov: bool
    extra: _ExtraInput

    def __init__(
        self,
        *,
        name: Literal["adamw", "adam", "sgd"] = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.9,
        nesterov: bool = False,
        extra: _ExtraInput = None,
    ) -> None:
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.momentum = momentum
        self.nesterov = nesterov
        self.extra = extra

    def to_pydantic(self) -> OptimizerConfig:
        """Build the frozen :class:`OptimizerConfig` from current state."""
        return OptimizerConfig(
            name=self.name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            momentum=self.momentum,
            nesterov=self.nesterov,
            extra=self.extra,  # type: ignore[arg-type]
        )


class SchedulerParams(BaseEstimator):
    """Mutable mirror of :class:`SchedulerConfig`."""

    name: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"]
    warmup_steps: int
    pct_start: float
    div_factor: float
    final_div_factor: float
    plateau_factor: float
    plateau_patience: int
    plateau_threshold: float
    min_lr: float
    extra: _ExtraInput

    def __init__(
        self,
        *,
        name: Literal[
            "constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"
        ] = "cosine_with_warmup",
        warmup_steps: int = 100,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        plateau_threshold: float = 1e-4,
        min_lr: float = 0.0,
        extra: _ExtraInput = None,
    ) -> None:
        self.name = name
        self.warmup_steps = warmup_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.min_lr = min_lr
        self.extra = extra

    def to_pydantic(self) -> SchedulerConfig:
        """Build the frozen :class:`SchedulerConfig` from current state."""
        return SchedulerConfig(
            name=self.name,
            warmup_steps=self.warmup_steps,
            pct_start=self.pct_start,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            plateau_factor=self.plateau_factor,
            plateau_patience=self.plateau_patience,
            plateau_threshold=self.plateau_threshold,
            min_lr=self.min_lr,
            extra=self.extra,  # type: ignore[arg-type]
        )


class LossParams(BaseEstimator):
    """Mutable mirror of :class:`LossConfig`.

    ``LossConfig.strategy`` has no pydantic default (it is task-type
    dependent; the estimator injects the task-appropriate value at build
    time). The sklearn adapter surface still needs a well-formed default
    so ``get_params`` / ``clone`` round-trip; ``"cross_entropy"`` is a
    placeholder the estimator overrides per task before ``to_pydantic``
    becomes authoritative.
    """

    strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    focal_gamma: float
    focal_alpha: float | None
    huber_delta: float
    label_smoothing: float
    extra: _ExtraInput

    def __init__(
        self,
        *,
        strategy: Literal[
            "cross_entropy", "focal", "mse", "mae", "huber", "pinball"
        ] = "cross_entropy",
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        huber_delta: float = 1.0,
        label_smoothing: float = 0.0,
        extra: _ExtraInput = None,
    ) -> None:
        self.strategy = strategy
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.huber_delta = huber_delta
        self.label_smoothing = label_smoothing
        self.extra = extra

    def to_pydantic(self) -> LossConfig:
        """Build the frozen :class:`LossConfig` from current state."""
        return LossConfig(
            strategy=self.strategy,
            focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha,
            huber_delta=self.huber_delta,
            label_smoothing=self.label_smoothing,
            extra=self.extra,  # type: ignore[arg-type]
        )


class SamplerParams(BaseEstimator):
    """Mutable mirror of :class:`SamplerConfig`."""

    strategy: Literal["none", "class_weighted", "oversample_minority", "undersample_majority"]
    oversample_ratio: float
    replacement: bool
    extra: _ExtraInput

    def __init__(
        self,
        *,
        strategy: Literal[
            "none", "class_weighted", "oversample_minority", "undersample_majority"
        ] = "none",
        oversample_ratio: float = 1.0,
        replacement: bool = True,
        extra: _ExtraInput = None,
    ) -> None:
        self.strategy = strategy
        self.oversample_ratio = oversample_ratio
        self.replacement = replacement
        self.extra = extra

    def to_pydantic(self) -> SamplerConfig:
        """Build the frozen :class:`SamplerConfig` from current state."""
        return SamplerConfig(
            strategy=self.strategy,
            oversample_ratio=self.oversample_ratio,
            replacement=self.replacement,
            extra=self.extra,  # type: ignore[arg-type]
        )


class TFTAdvancedParams(BaseEstimator):
    """Mutable mirror of :class:`TFTAdvancedConfig` (empty in v1 except extra)."""

    extra: _ExtraInput

    def __init__(self, *, extra: _ExtraInput = None) -> None:
        self.extra = extra

    def to_pydantic(self) -> TFTAdvancedConfig:
        """Build the frozen :class:`TFTAdvancedConfig` from current state."""
        return TFTAdvancedConfig(extra=self.extra)  # type: ignore[arg-type]
