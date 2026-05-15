"""Tier 1 optimizer sub-config (per architecture A4 / hyperparameter strategy).

Owns the optimizer name, its tunable defaults, and the ALPHA-tier
``extra`` escape hatch. The reserved-keys collision check lives here (the
config layer) so it surfaces at construction; ``build_optimizer``
(Phase 4) trusts the validated config and does not re-check.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._extras import ExtraDict

__all__ = ["OptimizerConfig"]


# Reserved set contains ONLY keys that collide with positional or typed
# kwargs at the build call site (the torch optimizer constructor). Untyped
# torch kwargs (maximize, foreach, capturable, differentiable, fused,
# dampening, etc.) are the legitimate ALPHA-tier passthrough use case and
# must NOT be blocked. When such a key is promoted to a typed field via
# ALPHA -> BETA, it moves into this reserved set and the typed field
# handles it.
_RESERVED_BY_OPTIMIZER: dict[str, frozenset[str]] = {
    "adamw": frozenset({"params", "lr", "weight_decay", "betas", "eps"}),
    "adam": frozenset({"params", "lr", "weight_decay", "betas", "eps"}),
    "sgd": frozenset({"params", "lr", "weight_decay", "momentum", "nesterov"}),
}


class OptimizerConfig(BaseModel):
    """Optimizer family sub-config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    # AdamW / Adam:
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    # SGD:
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = False
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        """Reject ``extra`` keys that collide with typed/positional build kwargs.

        Config-layer check (not build-time) so the error surfaces at
        construction.
        """
        reserved = _RESERVED_BY_OPTIMIZER[self.name]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.name} kwargs: "
                f"{sorted(clashes)}. Set the typed OptimizerConfig field "
                "directly."
            )
        return self
