"""Tier 1 imbalance-sampler sub-config (per architecture A4 / strategy doc).

The per-``strategy`` reserved set is the delegated design decision noted
in ``scheduler.py``: each entry lists the typed fields the corresponding
sampler branch consumes at build time.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._extras import ExtraDict

__all__ = ["SamplerConfig"]


_RESERVED_BY_SAMPLER: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "class_weighted": frozenset(),
    "oversample_minority": frozenset({"oversample_ratio", "replacement"}),
    "undersample_majority": frozenset({"replacement"}),
}


class SamplerConfig(BaseModel):
    """Imbalance-sampler family sub-config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["none", "class_weighted", "oversample_minority", "undersample_majority"] = (
        "none"
    )
    oversample_ratio: float = Field(default=1.0, gt=0.0)
    replacement: bool = True
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        """Reject ``extra`` keys that collide with this sampler's typed fields."""
        reserved = _RESERVED_BY_SAMPLER[self.strategy]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.strategy} kwargs: "
                f"{sorted(clashes)}. Set the typed SamplerConfig field "
                "directly."
            )
        return self
