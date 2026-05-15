"""Tier 1 LR-scheduler sub-config (per architecture A4 / strategy doc).

Schema copied verbatim from the authoritative source at
``docs/hyperparameter_strategy.md`` Tier 1; field menu and defaults
trace to requirements F5.

Design note: the strategy doc enumerates ``_RESERVED_BY_OPTIMIZER``
explicitly but specifies only that scheduler / loss / sampler carry
"analogous" reserved sets against their own typed fields. The
per-``name`` reserved sets below are that delegated design decision: each
entry lists the typed fields the corresponding scheduler branch consumes
at build time. The ``constant``-rejects-``warmup_steps`` interaction
(requirements F5 / arch I10) is intentionally NOT enforced here in
Phase 1; its enforcement layer is deferred to the Phase 4 design pass.
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
    """LR-scheduler family sub-config."""

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
