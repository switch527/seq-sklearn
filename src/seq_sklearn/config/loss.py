"""Tier 1 loss sub-config (per architecture A4 / strategy doc).

``strategy`` has no default: the legal value depends on ``task_type``,
and the F5 validity matrix on ``BaseModelConfig`` gates the
``(task_type, strategy)`` pair. The estimator's ``_build_config``
(Phase 6a) injects the task-appropriate default; Phase 1 keeps the field
required so ``LossConfig()`` raises.

The per-``strategy`` reserved set is the delegated design decision noted
in ``scheduler.py``: each entry lists the typed fields the corresponding
loss branch consumes at build time.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._extras import ExtraDict

__all__ = ["LossConfig"]


_RESERVED_BY_LOSS: dict[str, frozenset[str]] = {
    "cross_entropy": frozenset({"label_smoothing"}),
    "focal": frozenset({"focal_gamma", "focal_alpha"}),
    "mse": frozenset(),
    "mae": frozenset(),
    "huber": frozenset({"huber_delta"}),
    "pinball": frozenset(),
}


class LossConfig(BaseModel):
    """Loss family sub-config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    focal_gamma: float = Field(default=2.0, gt=0.0)
    focal_alpha: float | None = None
    huber_delta: float = Field(default=1.0, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        """Reject ``extra`` keys that collide with this loss's typed fields."""
        reserved = _RESERVED_BY_LOSS[self.strategy]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.strategy} kwargs: "
                f"{sorted(clashes)}. Set the typed LossConfig field "
                "directly."
            )
        return self
