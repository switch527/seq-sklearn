"""Loss-factory dispatch and numerical-stability tests (A8 / F5 / N1).

``build_loss`` must dispatch to the right concrete class per the F5
loss-class table, reject illegal combos with ``ConfigError``, and a
hypothesis property test asserts the loss output is a finite scalar for
every legal ``(task_type, loss_strategy)`` cell.
"""

import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings

from seq_sklearn.errors import ConfigError
from seq_sklearn.training.losses import (
    BinaryFocalLoss,
    MulticlassFocalLoss,
    PinballLoss,
    _ScalarOutputLoss,
    build_loss,
)

_QUANTILES = (0.1, 0.5, 0.9)


def _assert_finite_scalar(loss: torch.Tensor) -> None:
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def _build(task_type: str, loss_strategy: str, **kw: object) -> nn.Module:
    defaults: dict[str, object] = {
        "class_weights": None,
        "focal_gamma": 2.0,
        "huber_delta": 1.0,
        "quantiles": None,
    }
    defaults.update(kw)
    return build_loss(task_type, loss_strategy, **defaults)  # type: ignore[arg-type]


def _unwrap(loss: nn.Module) -> nn.Module:
    """The inner loss; binary / point losses are wrapped by the factory.

    ``build_loss`` wraps scalar-output losses in ``_ScalarOutputLoss``
    (the (B, 1) head -> (B,) loss bridge added in Phase 6a); the F5
    concrete-class contract is on the inner module.
    """
    return loss.inner if isinstance(loss, _ScalarOutputLoss) else loss


@pytest.mark.parametrize(
    ("task_type", "loss_strategy", "expected"),
    [
        ("binary", "cross_entropy", nn.BCEWithLogitsLoss),
        ("binary", "focal", BinaryFocalLoss),
        ("multiclass", "cross_entropy", nn.CrossEntropyLoss),
        ("multiclass", "focal", MulticlassFocalLoss),
        ("regression_point", "mse", nn.MSELoss),
        ("regression_point", "mae", nn.L1Loss),
        ("regression_point", "huber", nn.HuberLoss),
    ],
)
def test_dispatch_to_concrete_class(
    task_type: str, loss_strategy: str, expected: type[nn.Module]
) -> None:
    """Each legal cell builds the F5-mandated concrete class."""
    loss = _build(task_type, loss_strategy)
    assert isinstance(_unwrap(loss), expected)


def test_pinball_dispatch_with_quantiles() -> None:
    loss = _build("regression_quantile", "pinball", quantiles=_QUANTILES)
    assert isinstance(loss, PinballLoss)


def test_pinball_accepts_2d_target_without_unsqueeze() -> None:
    """Covers the already-2D target branch in PinballLoss.forward."""
    pinball = PinballLoss(quantiles=_QUANTILES)
    preds = torch.randn(4, len(_QUANTILES))
    target_2d = torch.randn(4, 1)
    _assert_finite_scalar(pinball(preds, target_2d))


def test_binary_class_weighted_sets_pos_weight() -> None:
    w = torch.tensor(3.0)
    loss = _unwrap(_build("binary", "cross_entropy", class_weights=w))
    assert isinstance(loss, nn.BCEWithLogitsLoss)
    assert loss.pos_weight is not None
    assert torch.equal(loss.pos_weight, w)


def test_multiclass_class_weighted_sets_weight() -> None:
    w = torch.tensor([1.0, 2.0, 3.0])
    loss = _build("multiclass", "cross_entropy", class_weights=w)
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.weight is not None
    assert torch.equal(loss.weight, w)


def test_v1_1_task_type_raises() -> None:
    with pytest.raises(ConfigError, match=r"scheduled for v1\.1"):
        _build("multilabel", "cross_entropy")


def test_illegal_task_loss_pair_raises() -> None:
    with pytest.raises(ConfigError, match=r"not legal for task_type='binary'"):
        _build("binary", "mse")


def test_focal_with_class_weights_raises() -> None:
    with pytest.raises(ConfigError, match=r"class_weights must be None"):
        _build("binary", "focal", class_weights=torch.tensor(2.0))


def test_class_weights_with_non_cross_entropy_raises() -> None:
    # Converse of the focal guard: F5 ties class weighting to
    # cross_entropy. Unreachable on legal v1 configs (check_combo
    # rejects first) but the factory boundary must defend it.
    with pytest.raises(ConfigError, match=r"only valid with loss_strategy='cross_entropy'"):
        _build("regression_point", "mse", class_weights=torch.ones(2))


def test_pinball_without_quantiles_raises() -> None:
    with pytest.raises(ConfigError, match=r"requires quantiles"):
        _build("regression_quantile", "pinball", quantiles=None)


@settings(deadline=None, max_examples=30)
@given(
    n=st.integers(min_value=1, max_value=8),
    num_classes=st.integers(min_value=2, max_value=5),
)
def test_loss_output_is_finite_scalar_every_legal_cell(n: int, num_classes: int) -> None:
    """Property: every legal cell yields a finite scalar on random logits."""
    g = torch.Generator().manual_seed(0)

    bin_logits = torch.randn(n, generator=g)
    bin_target = (torch.rand(n, generator=g) > 0.5).float()
    for strat in ("cross_entropy", "focal"):
        _assert_finite_scalar(_build("binary", strat)(bin_logits, bin_target))

    mc_logits = torch.randn(n, num_classes, generator=g)
    mc_target = torch.randint(0, num_classes, (n,), generator=g)
    for strat in ("cross_entropy", "focal"):
        _assert_finite_scalar(_build("multiclass", strat)(mc_logits, mc_target))

    reg_pred = torch.randn(n, 1, generator=g)
    reg_target = torch.randn(n, 1, generator=g)
    for strat in ("mse", "mae", "huber"):
        _assert_finite_scalar(_build("regression_point", strat)(reg_pred, reg_target))

    q_pred = torch.randn(n, len(_QUANTILES), generator=g)
    q_target = torch.randn(n, generator=g)
    pinball = _build("regression_quantile", "pinball", quantiles=_QUANTILES)
    _assert_finite_scalar(pinball(q_pred, q_target))
