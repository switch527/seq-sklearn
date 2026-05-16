"""LR-scheduler factory (per requirements F5 / architecture A7 / A20).

:func:`build_scheduler` consumes the validated :class:`SchedulerConfig`
plus the optimizer and returns the Lightning ``lr_scheduler`` dict
documented in A7 (``scheduler`` / ``interval`` / ``frequency`` /
``monitor`` / ``strict``). The Trainer curries it as
``partial(build_scheduler, config=cfg.scheduler, ...)`` and the
LightningModule splats the returned dict under the ``lr_scheduler`` key
of ``configure_optimizers``.

Two build-layer contracts the config layer deliberately does NOT
enforce (per the ``SchedulerConfig`` docstring) land here:

* ``constant`` with ``warmup_steps > 0`` raises :class:`ConfigError`
  (F5: ``constant`` ignores ``warmup_steps``; setting it is a mistake).
* ``one_cycle`` needs ``total_steps`` up front. Per A20 item 1 the
  explicit ``total_steps`` form is used (not ``epochs`` /
  ``steps_per_epoch``) so the schedule stays correct under
  ``accumulate_grad_batches``: Lightning steps the scheduler once per
  optimizer step, and the caller derives ``total_steps`` from the
  accumulation-adjusted optimizer-step count. ``total_steps`` is
  required whenever ``one_cycle`` is selected.
"""

import logging
import math
from collections.abc import Callable
from typing import Any

from torch import optim
from torch.optim.lr_scheduler import LRScheduler

from seq_sklearn.config._extras import extract_deprecated_extras
from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.errors import ConfigError

__all__ = ["build_scheduler"]

logger = logging.getLogger(__name__)


def _warmup_cosine_lambda(
    warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> Callable[[int], float]:
    """Build the LambdaLR multiplier: linear warmup then cosine decay.

    The multiplier is relative to the optimizer base LR; ``min_lr_ratio``
    is the floor as a fraction of the base LR so the configured absolute
    ``min_lr`` is honored without the scheduler reading the optimizer.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def build_scheduler(
    optimizer: optim.Optimizer,
    *,
    config: SchedulerConfig,
    monitor: str,
    total_steps: int | None = None,
) -> dict[str, object]:
    """Build the Lightning ``lr_scheduler`` dict named by ``config.name``.

    ``monitor`` is the validation metric name (mandatory for
    ``reduce_on_plateau``; carried on every dict so the shape is
    uniform). ``total_steps`` is the accumulation-adjusted optimizer-step
    count; required for ``one_cycle``, also used to bound the
    ``cosine_with_warmup`` decay horizon.

    Raises:
        ConfigError: ``constant`` was paired with ``warmup_steps > 0``,
            or a step-horizon scheduler (``one_cycle``,
            ``cosine_with_warmup``) was requested without ``total_steps``.
    """
    config, extra_typed = extract_deprecated_extras(config, "scheduler")
    # ALPHA passthrough is intentionally opaque kwargs; see optimizers.py.
    extra: dict[str, Any] = dict(extra_typed)

    if config.name == "constant":
        if config.warmup_steps > 0:
            raise ConfigError(
                "scheduler='constant' ignores warmup_steps; "
                f"got warmup_steps={config.warmup_steps}. Use "
                "'cosine_with_warmup' or 'one_cycle' for a warmup, or "
                "set warmup_steps=0."
            )
        scheduler: LRScheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _step: 1.0, **extra
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": monitor,
            "strict": True,
        }

    if config.name == "reduce_on_plateau":
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            min_lr=config.min_lr,
            **extra,
        )
        return {
            "scheduler": plateau,
            "interval": "epoch",
            "frequency": 1,
            "monitor": monitor,
            "strict": True,
        }

    if config.name == "one_cycle":
        if total_steps is None:
            raise ConfigError(
                "scheduler='one_cycle' requires total_steps; the Trainer "
                "derives it from the accumulation-adjusted optimizer-step "
                "count (A20 item 1)."
            )
        one_cycle = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
            **extra,
        )
        return {
            "scheduler": one_cycle,
            "interval": "step",
            "frequency": 1,
            "monitor": monitor,
            "strict": True,
        }

    # config.name == "cosine_with_warmup" (no other Literal member).
    if total_steps is None:
        raise ConfigError(
            "scheduler='cosine_with_warmup' requires total_steps to bound "
            "the decay horizon; the Trainer derives it from the "
            "accumulation-adjusted optimizer-step count."
        )
    # base_lr is always > 0: OptimizerConfig.learning_rate is gt=0.0 and
    # build_optimizer writes it into every param group's "lr".
    base_lr = optimizer.param_groups[0]["lr"]
    if config.min_lr >= base_lr:
        raise ConfigError(
            f"scheduler min_lr ({config.min_lr}) must be < the optimizer "
            f"learning_rate ({base_lr}) for cosine_with_warmup; otherwise "
            "the post-warmup multiplier climbs instead of decaying."
        )
    min_lr_ratio = config.min_lr / base_lr
    cosine = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine_lambda(config.warmup_steps, total_steps, min_lr_ratio),
        **extra,
    )
    return {
        "scheduler": cosine,
        "interval": "step",
        "frequency": 1,
        "monitor": monitor,
        "strict": True,
    }
