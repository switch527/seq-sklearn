"""Scheduler-factory dispatch and guard tests (F5 / A7 / A20).

Legal ``SchedulerConfig.name`` values build the matching concrete
scheduler wrapped in the Lightning ``lr_scheduler`` dict;
``constant`` with ``warmup_steps > 0`` raises ``ConfigError``;
``one_cycle`` / ``cosine_with_warmup`` without ``total_steps`` raise
``ConfigError``. The warmup-cosine LR multiplier is checked across its
phases.
"""

import pytest
import torch
from torch import optim

from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.errors import ConfigError
from seq_sklearn.training.schedulers import build_scheduler


def _opt(lr: float = 1e-3) -> optim.Optimizer:
    return optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def test_constant_builds_lambda_lr_epoch_interval() -> None:
    cfg = SchedulerConfig(name="constant", warmup_steps=0)
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss")
    assert isinstance(out["scheduler"], optim.lr_scheduler.LambdaLR)
    assert out["interval"] == "epoch"
    assert out["monitor"] == "val_loss"
    assert out["frequency"] == 1
    assert out["strict"] is True


def test_constant_with_warmup_steps_raises() -> None:
    cfg = SchedulerConfig(name="constant", warmup_steps=50)
    with pytest.raises(ConfigError, match=r"'constant' ignores warmup_steps"):
        build_scheduler(_opt(), config=cfg, monitor="val_loss")


def test_reduce_on_plateau_builds_plateau_epoch_interval() -> None:
    cfg = SchedulerConfig(name="reduce_on_plateau")
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss")
    assert isinstance(out["scheduler"], optim.lr_scheduler.ReduceLROnPlateau)
    assert out["interval"] == "epoch"
    assert out["monitor"] == "val_loss"


def test_one_cycle_builds_one_cycle_step_interval() -> None:
    cfg = SchedulerConfig(name="one_cycle")
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss", total_steps=100)
    assert isinstance(out["scheduler"], optim.lr_scheduler.OneCycleLR)
    assert out["interval"] == "step"


def test_one_cycle_without_total_steps_raises() -> None:
    cfg = SchedulerConfig(name="one_cycle")
    with pytest.raises(ConfigError, match=r"'one_cycle' requires total_steps"):
        build_scheduler(_opt(), config=cfg, monitor="val_loss")


def test_cosine_with_warmup_builds_lambda_lr_step_interval() -> None:
    cfg = SchedulerConfig(name="cosine_with_warmup", warmup_steps=10)
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss", total_steps=100)
    assert isinstance(out["scheduler"], optim.lr_scheduler.LambdaLR)
    assert out["interval"] == "step"


def test_cosine_with_warmup_without_total_steps_raises() -> None:
    cfg = SchedulerConfig(name="cosine_with_warmup")
    with pytest.raises(ConfigError, match=r"'cosine_with_warmup' requires total_steps"):
        build_scheduler(_opt(), config=cfg, monitor="val_loss")


def test_cosine_warmup_lr_multiplier_phases() -> None:
    """Warmup ramps linearly; post-warmup decays cosine toward min_lr."""
    cfg = SchedulerConfig(name="cosine_with_warmup", warmup_steps=10, min_lr=1e-4)
    base_lr = 1e-3
    out = build_scheduler(_opt(base_lr), config=cfg, monitor="val_loss", total_steps=100)
    sched = out["scheduler"]
    assert isinstance(sched, optim.lr_scheduler.LambdaLR)
    fn = sched.lr_lambdas[0]

    assert fn(0) == pytest.approx(0.0)
    assert fn(5) == pytest.approx(0.5)
    assert fn(10) == pytest.approx(1.0)
    # Midpoint of decay (step 55 of 100, halfway through the 90-step
    # decay): cosine factor 0.5, floored at min_lr/base_lr = 0.1.
    mid = fn(55)
    assert mid == pytest.approx(0.1 + 0.9 * 0.5)
    # End and beyond clamp to the min_lr ratio.
    assert fn(100) == pytest.approx(0.1)
    assert fn(200) == pytest.approx(0.1)


def test_cosine_warmup_zero_warmup_skips_ramp() -> None:
    """warmup_steps=0 short-circuits the warmup branch (step 0 -> decay)."""
    cfg = SchedulerConfig(name="cosine_with_warmup", warmup_steps=0)
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss", total_steps=50)
    sched = out["scheduler"]
    assert isinstance(sched, optim.lr_scheduler.LambdaLR)
    fn = sched.lr_lambdas[0]
    assert fn(0) == pytest.approx(1.0)


def test_cosine_warmup_degenerate_total_steps_le_warmup() -> None:
    """total_steps <= warmup_steps holds the multiplier at 1.0 post-warmup."""
    cfg = SchedulerConfig(name="cosine_with_warmup", warmup_steps=10)
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss", total_steps=10)
    sched = out["scheduler"]
    assert isinstance(sched, optim.lr_scheduler.LambdaLR)
    fn = sched.lr_lambdas[0]
    assert fn(10) == pytest.approx(1.0)


def test_extra_kwargs_pass_through_constant() -> None:
    cfg = SchedulerConfig(name="constant", warmup_steps=0, extra=(("last_epoch", -1),))
    out = build_scheduler(_opt(), config=cfg, monitor="val_loss")
    assert isinstance(out["scheduler"], optim.lr_scheduler.LambdaLR)
