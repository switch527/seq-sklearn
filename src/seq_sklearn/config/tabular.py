"""TabularToSequence pydantic configuration (per architecture A4 / F3).

Frozen model; the mutable :class:`TabularConfigParams` adapter at
:mod:`seq_sklearn.config._adapters` mirrors these fields for the
sklearn estimator surface.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

__all__ = ["TabularToSequenceConfig"]


def _normalize_embed_dims(v: Any) -> tuple[tuple[str, int], ...]:
    """Coerce dict / Mapping / tuple-of-tuples input to a sorted hashable form.

    Pydantic v2 frozen models must store hashable values to be hashable
    themselves; a plain ``dict`` in ``__dict__`` defeats hashing. Storing
    as a sorted ``tuple[tuple[str, int], ...]`` keeps the model hashable
    while letting callers pass a plain ``dict`` at construction time.
    Access via ``dict(cfg.categorical_embed_dims)`` round-trips the
    contents.
    """
    if v is None:
        return ()
    if isinstance(v, Mapping):
        items = list(v.items())
    elif isinstance(v, tuple):
        return tuple(sorted(v))
    else:
        items = list(v)
    return tuple(sorted(items))


CategoricalEmbedDims = Annotated[
    tuple[tuple[str, int], ...], BeforeValidator(_normalize_embed_dims)
]


class TabularToSequenceConfig(BaseModel):
    """Configuration for the panel-to-sequence preprocessing transformer.

    Frozen at the pydantic layer. ``categorical_embed_dims`` is stored as
    a sorted tuple of ``(column_name, embedding_dim)`` pairs so the
    frozen model is hashable; callers may pass a plain dict at
    construction time and the validator normalizes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id_col: str
    time_col: str
    static_categorical_cols: tuple[str, ...] = ()
    static_real_cols: tuple[str, ...] = ()
    time_varying_real_cols: tuple[str, ...] = ()
    time_varying_categorical_cols: tuple[str, ...] = ()
    lookback: int = Field(default=12, ge=1)
    prediction_step: int = Field(default=0, ge=0)
    min_periods: int = Field(default=1, ge=1)
    min_periods_predict: int = Field(default=1, ge=1)
    scaling_real: Literal["standard", "robust", "quantile_uniform", "none"] = "standard"
    scaling_static_real: Literal["standard", "robust", "quantile_uniform", "none", "inherit"] = (
        "inherit"
    )
    clip_features: float | None = None
    max_categorical_cardinality: int = Field(default=1000, ge=1)
    hash_high_cardinality: bool = False
    categorical_embed_dims: CategoricalEmbedDims = ()
