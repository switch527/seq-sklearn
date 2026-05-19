"""F1.1 post-fit attribute contract + A2 presence/absence (N1).

Pins the fitted-attribute table from requirements F1.1 on both family
bases, and the A2 rule that ``decision_threshold_`` / ``quantiles_`` are
ABSENT (``hasattr`` is False), not ``None``, when they do not apply.
CPU-pinned; tiny panel + 1 epoch so it stays a unit-speed test.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(
    target_kind: str = "binary", num_classes: int = 3
) -> tuple[SyntheticPanelGenerator, pd.DataFrame, np.ndarray]:
    gen = SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=16,
        periods_per_entity=18,
        signal_strength=0.7,
        lookback=6,
        num_classes=num_classes,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
    return gen, panel, y


def _tab(gen: SyntheticPanelGenerator) -> TabularConfigParams:
    return TabularConfigParams(
        id_col=gen.id_col,
        time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        min_periods=1,
        min_periods_predict=1,
        max_categorical_cardinality=10_000,
    )


def _declared(gen: SyntheticPanelGenerator) -> list[str]:
    return [
        *gen.static_categorical_cols,
        *gen.static_real_cols,
        *gen.time_varying_real_cols,
        *gen.time_varying_categorical_cols,
    ]


_COMMON: dict[str, Any] = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "hidden_size": 8,
    "attention_heads": 2,
    "max_epochs": 1,
    "batch_size": 32,
    "val_fraction": 0.2,
    "cal_fraction": 0.0,
    "precision": "32-true",
    "verbose": False,
    "seed": 42,
}


def _assert_shared_fit_state(est: object, gen: SyntheticPanelGenerator) -> None:
    declared = _declared(gen)
    assert est.n_features_in_ == len(declared)  # type: ignore[attr-defined]
    assert list(est.feature_names_in_) == declared  # type: ignore[attr-defined]
    assert isinstance(est.feature_names_in_, np.ndarray)  # type: ignore[attr-defined]
    assert est.n_outputs_ == 1  # type: ignore[attr-defined]
    assert hasattr(est, "config_")
    assert hasattr(est, "transformer_")
    assert hasattr(est, "feature_schema_fingerprint_")
    assert hasattr(est, "calibrator_")  # present (possibly None) when fitted


def test_classifier_binary_fit_state_and_threshold_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel("binary")
    # threshold_tuning needs a calibration fold (F2); override the
    # shared cal_fraction=0.0 so the tuner has data.
    est = TFTClassifier(
        task_type="binary",
        tabular_config=_tab(gen),
        threshold_tuning=True,
        **{**_COMMON, "cal_fraction": 0.2},
    ).fit(x, y)
    _assert_shared_fit_state(est, gen)
    assert isinstance(est.classes_, np.ndarray)
    assert list(est.classes_) == sorted(np.unique(y))
    assert est.n_classes_ == len(np.unique(y))
    # A2: binary + threshold_tuning -> decision_threshold_ PRESENT, float.
    assert hasattr(est, "decision_threshold_")
    assert isinstance(est.decision_threshold_, float)


def test_classifier_binary_without_tuning_threshold_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel("binary")
    est = TFTClassifier(
        task_type="binary", tabular_config=_tab(gen), threshold_tuning=False, **_COMMON
    ).fit(x, y)
    _assert_shared_fit_state(est, gen)
    # A2: ABSENT (not None) when threshold_tuning is off.
    assert not hasattr(est, "decision_threshold_")


def test_classifier_multiclass_threshold_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel("multiclass", num_classes=3)
    # A2: decision_threshold_ is absent for any non-binary classifier
    # regardless of threshold_tuning; keep it off so the test pins the
    # multiclass rule, not the cal-fold requirement.
    est = TFTClassifier(
        task_type="multiclass", tabular_config=_tab(gen), threshold_tuning=False, **_COMMON
    ).fit(x, y)
    _assert_shared_fit_state(est, gen)
    assert est.n_classes_ == 3
    # A2: non-binary classifier -> ABSENT even with threshold_tuning set.
    assert not hasattr(est, "decision_threshold_")


def test_point_regressor_fit_state_and_quantiles_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel("regression_point")
    est = TFTRegressor(task_type="regression_point", tabular_config=_tab(gen), **_COMMON).fit(
        x, y.astype(float)
    )
    _assert_shared_fit_state(est, gen)
    # A2 / F1.1: classes_ never on a regressor; quantiles_ ABSENT for point.
    assert not hasattr(est, "classes_")
    assert not hasattr(est, "quantiles_")
    assert not hasattr(est, "decision_threshold_")


def test_quantile_regressor_quantiles_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel("regression_point")
    est = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(gen),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(x, y.astype(float))
    _assert_shared_fit_state(est, gen)
    # F1.1: quantiles_ PRESENT (1D ndarray of fit-time quantiles) in
    # quantile mode.
    assert hasattr(est, "quantiles_")
    assert isinstance(est.quantiles_, np.ndarray)
    assert list(est.quantiles_) == [0.1, 0.5, 0.9]
    assert not hasattr(est, "decision_threshold_")
