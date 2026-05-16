"""Loss factory (per architecture A8 / requirements F5).

:func:`build_loss` maps a ``(task_type, loss_strategy)`` pair to a
concrete ``nn.Module`` per the F5 loss-class table. It is a string-in
boundary: callers in the LightningModule extract the string values from
the frozen pydantic config via the F5 bridge (``cfg.loss.strategy``,
``cfg.sampler.strategy``) AFTER routing ``cfg.loss`` through
``extract_deprecated_extras``. This factory is intentionally
post-extras (it takes resolved strings, not a ``LossConfig``), so the
ALPHA->BETA promotion contract is owned by that Phase 4b call site, not
here. Any pair not in the v1 validity matrix raises
:class:`ConfigError`; v1.1 task types raise with a "scheduled for v1.1"
message rather than failing later on shape mismatch.

``focal`` and ``pinball`` have no torch built-in, so this module ships
small modules for them. ``focal`` deliberately ignores ``class_weights``
(F5: selecting focal disables in-loss class weighting; imbalance moves
to the sampler side); the factory enforces ``class_weights is None``
when ``loss_strategy == "focal"``.
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as torch_functional

from seq_sklearn.config._domains import V1_1_TASK_TYPES
from seq_sklearn.config._validity import legal_task_loss_pairs
from seq_sklearn.errors import ConfigError

__all__ = ["BinaryFocalLoss", "MulticlassFocalLoss", "PinballLoss", "build_loss"]

logger = logging.getLogger(__name__)


# Legal v1 (task_type, loss_strategy) pairs, derived from the single
# _LEGAL_CELLS validity-matrix table so this factory cannot drift from
# the config-layer validator. Imbalance/calibration legality remains the
# config validator's job; the factory only needs (task, loss).
_LEGAL_TASK_LOSS: frozenset[tuple[str, str]] = legal_task_loss_pairs()


class BinaryFocalLoss(nn.Module):
    """Binary focal loss on raw logits.

    ``B`` = batch. Logits and targets are both ``(B,)`` or ``(B, 1)``;
    targets are float in ``{0, 1}``. No class weighting (F5: focal moves
    imbalance handling to the sampler).
    """

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return the mean focal loss over the batch."""
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        ce = torch_functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = ((1.0 - p_t) ** self.gamma) * ce
        return loss.mean()


class MulticlassFocalLoss(nn.Module):
    """Multi-class focal loss on raw logits.

    ``B`` = batch, ``K`` = num_classes. Logits are ``(B, K)``; targets
    are ``(B,)`` int64 class indices. No class weighting (F5).
    """

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return the mean focal loss over the batch."""
        log_prob = torch_functional.log_softmax(logits, dim=-1)
        ce = torch_functional.nll_loss(log_prob, target, reduction="none")
        p_t = log_prob.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
        loss = ((1.0 - p_t) ** self.gamma) * ce
        return loss.mean()


class PinballLoss(nn.Module):
    """Quantile (pinball) loss over a fixed quantile vector.

    ``B`` = batch, ``Q`` = number of quantiles. Predictions are
    ``(B, Q)``; targets are ``(B,)`` or ``(B, 1)`` and broadcast across
    the quantile axis.
    """

    quantiles: Tensor

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Return the mean pinball loss over the batch and quantiles."""
        if target.ndim == 1:
            target = target.unsqueeze(-1)  # (B,) -> (B, 1)
        error = target - preds  # (B, Q)
        q = self.quantiles  # (Q,)
        loss = torch.maximum(q * error, (q - 1.0) * error)
        return loss.mean()


def build_loss(
    task_type: str,
    loss_strategy: str,
    *,
    class_weights: Tensor | None,
    focal_gamma: float,
    huber_delta: float,
    quantiles: tuple[float, ...] | None,
) -> nn.Module:
    """Build the F5 loss module for a ``(task_type, loss_strategy)`` pair.

    ``class_weights`` is non-``None`` only when the sampler strategy is
    ``class_weighted``; it maps to ``BCEWithLogitsLoss(pos_weight=...)``
    for binary and ``CrossEntropyLoss(weight=...)`` for multiclass.

    Raises:
        ConfigError: the pair is not in the v1 validity matrix, a v1.1
            task type was passed, ``pinball`` was requested without
            ``quantiles``, or ``class_weights`` was supplied alongside a
            ``focal`` strategy (F5 forbids in-loss weighting for focal).
    """
    if task_type in V1_1_TASK_TYPES:
        raise ConfigError(
            f"task_type={task_type!r} is scheduled for v1.1; v1 supports "
            "single-output classification and regression only."
        )
    if (task_type, loss_strategy) not in _LEGAL_TASK_LOSS:
        legal = sorted(lt for tt, lt in _LEGAL_TASK_LOSS if tt == task_type)
        raise ConfigError(
            f"loss_strategy={loss_strategy!r} is not legal for "
            f"task_type={task_type!r}. Legal losses for this task: {legal}."
        )
    if loss_strategy == "focal" and class_weights is not None:
        raise ConfigError(
            "class_weights must be None when loss_strategy='focal': F5 "
            "moves imbalance handling to the sampler side for focal."
        )

    if task_type == "binary" and loss_strategy == "cross_entropy":
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    if task_type == "multiclass" and loss_strategy == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_strategy == "focal":
        if task_type == "binary":
            return BinaryFocalLoss(gamma=focal_gamma)
        return MulticlassFocalLoss(gamma=focal_gamma)
    if task_type == "regression_point":
        if loss_strategy == "mse":
            return nn.MSELoss()
        if loss_strategy == "mae":
            return nn.L1Loss()
        return nn.HuberLoss(delta=huber_delta)
    # Only (regression_quantile, pinball) remains.
    if quantiles is None:
        raise ConfigError(
            "loss_strategy='pinball' requires quantiles to be set (regression_quantile task)."
        )
    return PinballLoss(quantiles=quantiles)
