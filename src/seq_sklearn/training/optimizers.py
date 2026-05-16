"""Optimizer factory (per requirements F5 / architecture A7).

:func:`build_optimizer` consumes the validated, frozen
:class:`OptimizerConfig` and the model parameter iterable, returning a
concrete ``torch.optim.Optimizer``. The Trainer curries it as
``partial(build_optimizer, config=cfg.optimizer)`` and the
LightningModule calls ``optimizer_factory(self.parameters())`` per the
A7 construction order.

Reserved-key collisions are already rejected at config construction
(``OptimizerConfig._check_extra_not_reserved``); this factory trusts
the validated config and only routes ALPHA ``extra`` keys through
:func:`extract_deprecated_extras` before splatting them as ``**extra``.
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import optim

from seq_sklearn.config._extras import extract_deprecated_extras
from seq_sklearn.config.optimizer import OptimizerConfig

__all__ = ["build_optimizer"]

logger = logging.getLogger(__name__)


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    config: OptimizerConfig,
) -> optim.Optimizer:
    """Build the F5 optimizer named by ``config.name``.

    ``adamw`` (the default) applies decoupled weight decay; ``adam``
    uses the same betas/eps without decoupling; ``sgd`` uses
    ``config.momentum`` / ``config.nesterov`` for a non-adaptive
    baseline. ALPHA ``extra`` kwargs pass straight through to the torch
    constructor.
    """
    config, extra_typed = extract_deprecated_extras(config, "optimizer")
    # The `extra` ALPHA passthrough is intentionally opaque kwargs: its
    # values are the JSON-safe ExtraValue union and the torch optimizer
    # constructors carry stricter per-parameter types. Binding to
    # dict[str, Any] preserves the runtime splat contract under strict
    # pyright (the config layer already rejected reserved-key clashes).
    extra: dict[str, Any] = dict(extra_typed)
    if config.name == "adamw":
        return optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
            **extra,
        )
    if config.name == "adam":
        return optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
            **extra,
        )
    # config.name == "sgd" (the Literal domain has no other member).
    return optim.SGD(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        nesterov=config.nesterov,
        **extra,
    )
