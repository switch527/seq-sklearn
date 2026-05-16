"""Tier 1 LR-scheduler sub-config (per architecture A4, requirements F5).

``_RESERVED_BY_SCHEDULER`` keys each scheduler name to the typed fields
its build branch consumes; an ``extra`` key colliding with one of those
is rejected at construction by :meth:`SchedulerConfig._check_extra_not_reserved`.
The ``constant``-rejects-``warmup_steps`` interaction (requirements F5 /
arch I10) is a build-layer concern and is not enforced at this config
layer; a build-time guard rejects that combination when the scheduler
factory is implemented.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._extras import ExtraDict

__all__ = ["SchedulerConfig"]


_RESERVED_BY_SCHEDULER: dict[str, frozenset[str]] = {
    "constant": frozenset(),
    "cosine_with_warmup": frozenset({"warmup_steps", "min_lr"}),
    "one_cycle": frozenset({"warmup_steps", "pct_start", "div_factor", "final_div_factor"}),
    "reduce_on_plateau": frozenset({"plateau_factor", "plateau_patience", "plateau_threshold"}),
}


class SchedulerConfig(BaseModel):
    """Frozen; the mutable sklearn surface is :class:`SchedulerParams`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"] = (
        "cosine_with_warmup"
    )
    warmup_steps: int = Field(default=100, ge=0)
    # OneCycleLR:
    pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)
    div_factor: float = Field(default=25.0, gt=0.0)
    final_div_factor: float = Field(default=1e4, gt=0.0)
    # ReduceLROnPlateau:
    plateau_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    plateau_patience: int = Field(default=5, ge=1)
    plateau_threshold: float = Field(default=1e-4, gt=0.0)
    # Cosine:
    min_lr: float = Field(default=0.0, ge=0.0)
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        """Reject ``extra`` keys that collide with this scheduler's typed fields."""
        reserved = _RESERVED_BY_SCHEDULER[self.name]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.name} kwargs: "
                f"{sorted(clashes)}. Set the typed SchedulerConfig field "
                "directly."
            )
        return self
