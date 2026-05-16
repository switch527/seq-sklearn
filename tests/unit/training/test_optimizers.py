"""Optimizer-factory dispatch tests (per requirements F5).

Each legal ``OptimizerConfig.name`` builds the matching concrete
``torch.optim`` class with the configured hyperparameters; ALPHA
``extra`` kwargs pass through to the constructor.
"""

import pytest
import torch
from torch import optim

from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.training.optimizers import build_optimizer


def _params() -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.zeros(3))]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("adamw", optim.AdamW),
        ("adam", optim.Adam),
        ("sgd", optim.SGD),
    ],
)
def test_dispatch_to_concrete_optimizer(name: str, expected: type[optim.Optimizer]) -> None:
    cfg = OptimizerConfig(name=name)  # type: ignore[arg-type]
    opt = build_optimizer(_params(), config=cfg)
    assert isinstance(opt, expected)


def test_adamw_carries_configured_hyperparameters() -> None:
    cfg = OptimizerConfig(
        name="adamw",
        learning_rate=3e-4,
        weight_decay=0.05,
        betas=(0.8, 0.95),
        eps=1e-7,
    )
    opt = build_optimizer(_params(), config=cfg)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(0.05)
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == pytest.approx(1e-7)


def test_adam_carries_configured_hyperparameters() -> None:
    cfg = OptimizerConfig(name="adam", learning_rate=1e-2, betas=(0.7, 0.9))
    opt = build_optimizer(_params(), config=cfg)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(1e-2)
    assert group["betas"] == (0.7, 0.9)


def test_sgd_uses_momentum_and_nesterov() -> None:
    cfg = OptimizerConfig(name="sgd", learning_rate=0.1, momentum=0.8, nesterov=True)
    opt = build_optimizer(_params(), config=cfg)
    group = opt.param_groups[0]
    assert group["momentum"] == pytest.approx(0.8)
    assert group["nesterov"] is True


def test_extra_kwargs_pass_through() -> None:
    """ALPHA `extra` keys reach the torch constructor (e.g. amsgrad)."""
    cfg = OptimizerConfig(name="adamw", extra=(("amsgrad", True),))
    opt = build_optimizer(_params(), config=cfg)
    assert opt.param_groups[0]["amsgrad"] is True
