"""Trainer construction / wiring tests (per architecture A7 / F5 / N4).

The Trainer is exercised against a stub transformer (the only true
external boundary per the testing conventions) that yields a synthetic
window dict the Phase 4a ``_DummyBackbone`` consumes. The tests pin the
A7 callback list and precision wiring, the N4 deterministic gate, the
F5 DataLoader defaults, the A20-item-6 sampler dispatch, the A20-item-1
``total_steps`` derivation, and the A20-item-5 ``resume_path`` ->
``ckpt_path`` threading with :class:`RngStateCallback` restore on resume.
"""

import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, nn

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.errors import ConfigError
from seq_sklearn.hardware import HardwareTier
from seq_sklearn.training.callbacks import (
    EventEmitter,
    GradScalerWatchdog,
    RngStateCallback,
)
from seq_sklearn.training.trainer import Trainer, _TensorDictDataset
from tests._test_models._dummy_modules import _DummyBackbone, _DummyHead


class _StubTransformer:
    """Stand-in for a fitted ``TabularToSequence``.

    ``transform`` returns a synthetic window dict for two entities (four
    windows each) so :func:`compute_three_way_split` has a non-trivial
    per-entity time axis and ``_DummyBackbone`` has a ``features`` key.
    """

    def __init__(self, n_per_entity: int = 4) -> None:
        self._n = n_per_entity

    def transform(self, x_panel: object) -> dict[str, Tensor]:
        del x_panel
        n = self._n
        total = 2 * n
        return {
            "features": torch.randn(total, 4),
            "target": torch.cat([torch.zeros(n, 1), torch.ones(n, 1)]),
            "entity_id": torch.tensor([0] * n + [1] * n, dtype=torch.long),
        }


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin hardware detection / CUDA off so tests are host-independent.

    GPU detection is a true external boundary (testing conventions allow
    mocking it); the dev host has a GPU but CI is CPU-only, so the
    precision / pin_memory / accelerator assertions must not depend on
    the runner.
    """
    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _constant_scheduler() -> SchedulerConfig:
    """``constant`` with ``warmup_steps=0`` (constant rejects warmup > 0)."""
    return SchedulerConfig(name="constant", warmup_steps=0)


def _model_factory() -> tuple[nn.Module, nn.Module]:
    return _DummyBackbone(), _DummyHead()


def _config(**overrides: object) -> BaseModelConfig:
    base: dict[str, object] = {
        "task_type": "binary",
        "loss": LossConfig(strategy="cross_entropy"),
        "batch_size": 2,
        "max_epochs": 1,
        "val_fraction": 0.25,
        "cal_fraction": 0.0,
        "num_workers": 0,
    }
    base.update(overrides)
    return BaseModelConfig(**base)  # type: ignore[arg-type]


# --- pl.Trainer construction: callbacks + precision --------------------


def test_build_pl_trainer_attaches_a7_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    pl_trainer = trainer.build_pl_trainer("32-true")

    kinds = {type(cb) for cb in pl_trainer.callbacks}
    for expected in (
        EarlyStopping,
        ModelCheckpoint,
        GradScalerWatchdog,
        EventEmitter,
        RngStateCallback,
    ):
        assert expected in kinds
    # F9 NaN-skip moved into _LightningModule.training_step; no
    # NaNLossGuard callback (a post-hoc callback cannot skip the step).
    from seq_sklearn.training import callbacks as _cb

    assert not hasattr(_cb, "NaNLossGuard")
    assert pl_trainer.logger is None
    assert "32-true" in str(pl_trainer.precision_plugin.precision)


def test_resolve_precision_auto_on_cpu_is_32_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    assert trainer._resolve_precision() == "32-true"


def test_resolve_precision_explicit_passthrough() -> None:
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        _config(precision="bf16-mixed"),
        _model_factory,
    )
    assert trainer._resolve_precision() == "bf16-mixed"


# --- N4 deterministic gate ---------------------------------------------


def test_deterministic_gate_on_32_true(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    called: list[bool] = []
    monkeypatch.setattr(
        "seq_sklearn.training.trainer.enable_strict_mode",
        lambda: called.append(True),
    )
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    trainer.build_pl_trainer("32-true")
    # enable_strict_mode is stubbed to a no-op here, so a True torch
    # deterministic flag isolates Lightning's own deterministic=True
    # wiring (proving the gate threaded it into pl.Trainer); the
    # strict_mode_globals autouse fixture restores it at teardown.
    assert torch.are_deterministic_algorithms_enabled() is True
    assert trainer._deterministic("32-true") is True
    assert called == [True]


def test_deterministic_gate_off_on_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    called: list[bool] = []
    monkeypatch.setattr(
        "seq_sklearn.training.trainer.enable_strict_mode",
        lambda: called.append(True),
    )
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    trainer.build_pl_trainer("bf16-mixed")
    assert trainer._deterministic("bf16-mixed") is False
    assert called == []


def test_deterministic_gate_off_on_32_not_true() -> None:
    # Pins the "32-true" literal against the near-miss "32": the gate
    # must not relax to a prefix / substring match.
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    assert trainer._deterministic("32") is False


# --- F5 DataLoader defaults --------------------------------------------


def test_dataloader_kwargs_defaults_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        _config(num_workers=None, pin_memory=None),
        _model_factory,
    )
    kw = trainer._dataloader_kwargs()
    assert kw["pin_memory"] is False  # CUDA forced off
    assert kw["persistent_workers"] == (kw["num_workers"] > 0)  # type: ignore[operator]


def test_dataloader_kwargs_pin_memory_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        _config(num_workers=None, pin_memory=None),
        _model_factory,
    )
    assert trainer._dataloader_kwargs()["pin_memory"] is True


def test_dataloader_kwargs_overrides_respected() -> None:
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        _config(num_workers=0, pin_memory=True),
        _model_factory,
    )
    kw = trainer._dataloader_kwargs()
    assert kw["num_workers"] == 0
    assert kw["pin_memory"] is True
    assert kw["persistent_workers"] is False


# --- total_steps derivation (A20 item 1) -------------------------------


def test_total_steps_none_for_constant_scheduler() -> None:
    cfg = _config(scheduler=SchedulerConfig(name="constant"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    assert trainer._total_steps(8) is None


def test_total_steps_derived_for_step_scheduler() -> None:
    cfg = _config(
        scheduler=SchedulerConfig(name="cosine_with_warmup"),
        max_epochs=3,
        batch_size=4,
        accumulate_grad_batches=2,
    )
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # 10 windows -> ceil(10/4)=3 batches -> ceil(3/2)=2 opt steps -> 3*2.
    assert trainer._total_steps(10) == 6


def test_total_steps_empty_train_fold_raises() -> None:
    cfg = _config(scheduler=SchedulerConfig(name="one_cycle"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match=r"training\s+fold is empty"):
        trainer._total_steps(0)


# --- sampler dispatch (A20 item 6) -------------------------------------


def test_train_sampler_none_for_default_strategy() -> None:
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    out = trainer._train_sampler(np.arange(4), torch.zeros(4, 1))
    assert out is None


def test_train_sampler_oversample_builds_subset_sampler() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="oversample_minority"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    targets = torch.tensor([0, 0, 0, 1]).reshape(-1, 1)
    sampler = trainer._train_sampler(np.arange(4), targets)
    assert sampler is not None
    assert len(list(sampler)) >= 4


def test_train_sampler_undersample_builds_subset_sampler() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="undersample_majority"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    targets = torch.tensor([0, 0, 0, 1]).reshape(-1, 1)
    sampler = trainer._train_sampler(np.arange(4), targets)
    assert sampler is not None
    assert len(list(sampler)) == 2  # both classes cut to the minority (1)


# --- window_time_index reconstruction ----------------------------------


def test_window_time_index_per_entity_ordinals() -> None:
    eids = np.array([0, 0, 0, 1, 1])
    np.testing.assert_array_equal(Trainer._window_time_index(eids), np.array([0, 1, 2, 0, 1]))


def test_window_time_index_empty() -> None:
    assert Trainer._window_time_index(np.array([], dtype=int)).size == 0


def test_window_time_index_non_monotone_raises() -> None:
    # A future non-TTS caller passing an unordered entity_id must fail
    # loudly here instead of silently corrupting the fold ordinals.
    eids = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="monotone"):
        Trainer._window_time_index(eids)


# --- _TensorDictDataset ------------------------------------------------


def test_tensor_dict_dataset_indexes_per_row() -> None:
    ds = _TensorDictDataset({"a": torch.arange(6).reshape(3, 2)})
    assert len(ds) == 3
    assert torch.equal(ds[1]["a"], torch.tensor([2, 3]))


# --- _configure_loss routes through extract_deprecated_extras ----------


def test_configure_loss_builds_bce_for_binary_cross_entropy() -> None:
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    loss = trainer._configure_loss(None)
    assert isinstance(loss, nn.BCEWithLogitsLoss)
    assert loss.pos_weight is None


def test_configure_loss_warns_on_unrecognized_loss_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(loss=LossConfig(strategy="cross_entropy", extra={"unknown_key": "v"}))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    with caplog.at_level("WARNING", logger="seq_sklearn.training.trainer"):
        trainer._configure_loss(None)
    assert any(
        "unknown_key" in r.message and "loss.extra" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


# --- F5 class_weighted: weights derived from the train fold ------------


def test_class_weights_none_for_default_strategy() -> None:
    trainer = Trainer(_StubTransformer(), _config(), _model_factory)  # type: ignore[arg-type]
    assert trainer._class_weights(np.arange(4), torch.zeros(4, 1)) is None


def test_class_weighted_binary_sets_pos_weight_from_fold() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Imbalanced fold: 6 negatives, 2 positives -> pos_weight = 6/2 = 3.
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32).reshape(-1, 1)
    weights = trainer._class_weights(np.arange(8), targets)
    assert weights is not None
    assert torch.allclose(weights, torch.tensor([3.0]))
    loss = trainer._configure_loss(weights)
    assert isinstance(loss, nn.BCEWithLogitsLoss)
    assert loss.pos_weight is not None
    assert torch.allclose(loss.pos_weight, torch.tensor([3.0]))


def test_class_weighted_multiclass_sets_non_uniform_weight() -> None:
    cfg = _config(task_type="multiclass", sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # 3-class imbalanced fold: counts [4, 2, 2] -> inverse-frequency weights.
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.int64)
    weights = trainer._class_weights(np.arange(8), targets)
    assert weights is not None
    assert weights.numel() == 3
    assert not torch.allclose(weights, weights[0].expand_as(weights))
    expected = torch.tensor([8.0 / 4.0, 8.0 / 2.0, 8.0 / 2.0])
    assert torch.allclose(weights, expected)
    loss = trainer._configure_loss(weights)
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.weight is not None
    assert torch.allclose(loss.weight, expected)


# --- F5 no-leakage: strict-subset train_idx mutation pins --------------
#
# These assert the production logic slices by train_idx (it does); they
# FAIL if `targets[train_idx]` is replaced with `targets`. Every other
# class-weight / sampler test passes train_idx = arange(N) so the slice
# is an identity no-op and the mutation survives the rest of the suite.


def test_class_weights_binary_uses_only_train_fold_balance() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Full panel: 4 pos / 4 neg -> pos_weight 1.0. Train fold (strict
    # subset) sees 6 neg / 2 pos -> pos_weight 3.0; the held-out tail is
    # all-positive. The slice MUST restrict to the fold.
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32).reshape(-1, 1)
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # 6 neg, 2 pos
    weights = trainer._class_weights(train_idx, targets)
    assert weights is not None
    assert torch.allclose(weights, torch.tensor([3.0]))  # fold-restricted
    full = trainer._class_weights(np.arange(12), targets)
    assert full is not None
    assert torch.allclose(full, torch.tensor([1.0]))  # differs from fold


def test_class_weights_multiclass_uses_only_train_fold_balance() -> None:
    cfg = _config(task_type="multiclass", sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Full panel counts [3, 3, 3]; train fold (strict subset) counts
    # [4, 1, 1]. Fold-restricted weights differ from the panel weights.
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=torch.int64)
    train_idx = np.array([0, 1, 2, 9, 4, 7])  # classes: 0,0,0,0,1,2
    weights = trainer._class_weights(train_idx, targets)
    assert weights is not None
    expected_fold = torch.tensor([6.0 / 4.0, 6.0 / 1.0, 6.0 / 1.0])
    assert torch.allclose(weights, expected_fold)
    full = trainer._class_weights(np.arange(12), targets)
    assert full is not None
    assert not torch.allclose(weights, full)  # fold balance, not panel


def test_train_sampler_undersample_uses_only_train_fold_balance() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="undersample_majority"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Full panel: 6 neg / 6 pos. Train fold (strict subset) is 5 neg /
    # 1 pos, so undersampling cuts to the fold minority (1 per class -> 2
    # indices, all drawn from train_idx). The panel balance would give 12.
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=torch.int64).reshape(-1, 1)
    train_idx = np.array([0, 1, 2, 3, 4, 5])  # 5 neg, 1 pos
    sampler = trainer._train_sampler(train_idx, targets)
    assert sampler is not None
    drawn = list(sampler)
    assert len(drawn) == 2  # fold minority (1) per class, not panel (6)
    assert set(drawn).issubset(set(train_idx.tolist()))


def test_train_sampler_oversample_uses_only_train_fold_balance() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="oversample_minority"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Train fold (strict subset) is 4 neg / 1 pos; oversampling lifts the
    # minority to the fold majority (8 total), every index in train_idx.
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int64).reshape(-1, 1)
    train_idx = np.array([0, 1, 2, 3, 4])  # 4 neg, 1 pos
    sampler = trainer._train_sampler(train_idx, targets)
    assert sampler is not None
    drawn = list(sampler)
    assert len(drawn) == 8  # fold-balanced (4+4), not panel-balanced
    assert set(drawn).issubset(set(train_idx.tolist()))


# --- CRITICAL 2: multiclass vocabulary sized from the full targets -----


def test_class_weights_multiclass_vocab_from_full_not_train_fold() -> None:
    cfg = _config(task_type="multiclass", sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    # Class 2 is held entirely out of the train fold. Sizing the vocab
    # from train_idx alone would yield a length-2 vector and an opaque
    # CrossEntropyLoss shape mismatch on the first forward.
    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    train_idx = np.array([0, 1, 2, 3])  # classes 0,0,1,1 (no class 2)
    weights = trainer._class_weights(train_idx, targets)
    assert weights is not None
    assert weights.numel() == 3  # full K, not the fold's 2
    # class 2 has zero fold count -> clamp(min=1.0) -> sum/1 weight.
    assert torch.allclose(weights, torch.tensor([4.0 / 2.0, 4.0 / 2.0, 4.0 / 1.0]))


# --- IMPROVEMENT 3: degenerate binary fold falls back to 1.0 -----------


def test_class_weights_binary_all_positive_fold_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    targets = torch.ones(6, 1, dtype=torch.float32)  # no negatives
    with caplog.at_level("WARNING", logger="seq_sklearn.training.trainer"):
        weights = trainer._class_weights(np.arange(6), targets)
    assert weights is not None
    assert torch.allclose(weights, torch.tensor([1.0]))
    assert any("no negative" in r.message and r.levelname == "WARNING" for r in caplog.records)


def test_class_weights_binary_all_negative_fold_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    targets = torch.zeros(6, 1, dtype=torch.float32)  # no positives
    with caplog.at_level("WARNING", logger="seq_sklearn.training.trainer"):
        weights = trainer._class_weights(np.arange(6), targets)
    assert weights is not None
    assert torch.allclose(weights, torch.tensor([1.0]))
    assert any("no positive" in r.message and r.levelname == "WARNING" for r in caplog.records)


# --- A16: optuna_trial threads from Trainer.fit into _LightningModule --


def test_fit_threads_optuna_trial_and_prunes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_cpu(monkeypatch)
    monkeypatch.chdir(tmp_path)
    trial = optuna.trial.FixedTrial({})
    monkeypatch.setattr(trial, "should_prune", lambda: True)
    cfg = _config(scheduler=_constant_scheduler())
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    with pytest.raises(optuna.TrialPruned):
        trainer.fit(object(), optuna_trial=trial)


# --- named-branch pins (qa-sonnet IMPROVEMENT 9) -----------------------


def test_train_sampler_none_for_class_weighted_strategy() -> None:
    cfg = _config(sampler=SamplerConfig(strategy="class_weighted"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    assert trainer._train_sampler(np.arange(4), torch.zeros(4, 1)) is None


def test_total_steps_none_for_reduce_on_plateau_scheduler() -> None:
    cfg = _config(scheduler=SchedulerConfig(name="reduce_on_plateau"))
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    assert trainer._total_steps(8) is None


# --- end-to-end fit + resume threading ---------------------------------


def test_fit_runs_one_epoch_and_returns_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_cpu(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = _config(scheduler=_constant_scheduler())
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    module = trainer.fit(object())
    assert module._last_train_output is not None


def test_fit_passes_train_idx_not_full_panel_to_class_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins the fit() call-site wiring: the index handed to
    # _class_weights must be the train fold (a strict subset), never
    # arange(N) over the full panel, so val/cal class balance cannot
    # leak into the loss weighting (A8 architecture.md:1208).
    _force_cpu(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = _config(
        scheduler=_constant_scheduler(),
        sampler=SamplerConfig(strategy="class_weighted"),
    )
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    seen: dict[str, object] = {}
    original = Trainer._class_weights

    def _spy(self: Trainer, train_idx: np.ndarray, targets: object) -> object:
        seen["idx"] = np.asarray(train_idx).copy()
        seen["n_total"] = int(np.asarray(targets).reshape(-1).shape[0])
        return original(self, train_idx, targets)  # type: ignore[arg-type]

    monkeypatch.setattr(Trainer, "_class_weights", _spy)
    trainer.fit(object())

    idx = seen["idx"]
    n_total = seen["n_total"]
    assert isinstance(idx, np.ndarray)
    assert isinstance(n_total, int)
    assert len(idx) < n_total  # strict subset: val/cal held out
    assert not np.array_equal(idx, np.arange(n_total))


def test_fit_with_oversample_sampler_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = _config(
        scheduler=_constant_scheduler(),
        sampler=SamplerConfig(strategy="oversample_minority"),
    )
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    module = trainer.fit(object())
    assert module is not None


def test_resume_path_threads_to_ckpt_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_fit(
        self: pl.Trainer,
        model: object,
        train: object,
        val: object,
        ckpt_path: object = None,
    ) -> None:
        del self, model, train, val
        captured["ckpt_path"] = ckpt_path

    _force_cpu(monkeypatch)
    monkeypatch.setattr(pl.Trainer, "fit", fake_fit)
    cfg = _config(scheduler=_constant_scheduler())
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        cfg,
        _model_factory,
        resume_path="/tmp/run.ckpt",
    )
    trainer.fit(object())
    assert captured["ckpt_path"] == "/tmp/run.ckpt"


def test_resume_path_path_object_threads_to_ckpt_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fit(
        self: pl.Trainer,
        model: object,
        train: object,
        val: object,
        ckpt_path: object = None,
    ) -> None:
        del self, model, train, val
        captured["ckpt_path"] = ckpt_path

    _force_cpu(monkeypatch)
    monkeypatch.setattr(pl.Trainer, "fit", fake_fit)
    cfg = _config(scheduler=_constant_scheduler())
    trainer = Trainer(
        _StubTransformer(),  # type: ignore[arg-type]
        cfg,
        _model_factory,
        resume_path=Path("/tmp/run.ckpt"),
    )
    trainer.fit(object())
    assert captured["ckpt_path"] == "/tmp/run.ckpt"  # str() of the Path


def test_no_resume_path_passes_none_ckpt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_fit(
        self: pl.Trainer,
        model: object,
        train: object,
        val: object,
        ckpt_path: object = None,
    ) -> None:
        del self, model, train, val
        captured["ckpt_path"] = ckpt_path

    _force_cpu(monkeypatch)
    monkeypatch.setattr(pl.Trainer, "fit", fake_fit)
    cfg = _config(scheduler=_constant_scheduler())
    trainer = Trainer(_StubTransformer(), cfg, _model_factory)  # type: ignore[arg-type]
    trainer.fit(object())
    assert captured["ckpt_path"] is None


def test_rng_state_callback_restores_on_resume() -> None:
    """The Trainer's RngStateCallback round-trips RNG state on resume.

    Asserts the A20-item-5 contract: after ``RngStateCallback`` (the
    callback the Trainer attaches) round-trips through the checkpoint,
    the Python / numpy / torch RNG streams match the byte-exact state
    captured at save time (Lightning's own restore has the issue-20204
    gap this closes).
    """
    cb = RngStateCallback()
    trainer = pl.Trainer(logger=False)
    module = pl.LightningModule()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    checkpoint: dict[str, object] = {}
    cb.on_save_checkpoint(trainer, module, checkpoint)
    expected = (random.random(), float(np.random.random()), torch.rand(1).item())

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    cb.on_load_checkpoint(trainer, module, checkpoint)
    restored = (random.random(), float(np.random.random()), torch.rand(1).item())
    assert restored == expected
