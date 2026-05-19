"""N1 concrete acceptance thresholds (slow, nightly).

Three-seed median on the F6 DGP at dgp_version=1, signal_strength=0.7,
seed triple (42, 137, 9999), full lookback 12, n~=2000 windows:

* binary classifier accuracy >= 0.75
* multiclass (4-class) macro-F1 >= 0.60
* point regressor R^2 >= 0.5
* quantile regressor coverage on the nominal 80% interval in
  [0.75, 0.85] after conformal calibration

Marked ``slow`` so it is excluded from the 5-minute per-PR budget (N2)
and runs nightly. CPU-pinned. The model-capacity / epoch knobs are the
deterministic starting point; the authoritative values are pinned to
dgp_version=1 and re-validated whenever the DGP version is bumped (N1).
"""

import statistics
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_SEEDS = (42, 137, 9999)


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _gen(target_kind: str, seed: int, num_classes: int = 3) -> SyntheticPanelGenerator:
    return SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=200,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=12,
        num_classes=num_classes,
        seed=seed,
    )


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


def _common(seed: int) -> dict[str, Any]:
    return {
        "scheduler": SchedulerParams(name="constant", warmup_steps=0),
        "hidden_size": 64,
        "attention_heads": 4,
        "max_epochs": 40,
        "batch_size": 128,
        "val_fraction": 0.2,
        "precision": "32-true",
        "verbose": False,
        "seed": seed,
    }


def _median_over_seeds(metric: Callable[[int], float]) -> float:
    return statistics.median(metric(s) for s in _SEEDS)


def test_binary_classifier_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)

    def acc(seed: int) -> float:
        gen = _gen("binary", seed)
        x, y = gen.generate(seed=seed)
        est = TFTClassifier(
            task_type="binary", tabular_config=_tab(gen), cal_fraction=0.0, **_common(seed)
        ).fit(x, y)
        return float(accuracy_score(y, est.predict(x)))

    assert _median_over_seeds(acc) >= 0.75


def test_multiclass_macro_f1(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)

    def macro_f1(seed: int) -> float:
        gen = _gen("multiclass", seed, num_classes=4)
        x, y = gen.generate(seed=seed)
        est = TFTClassifier(
            task_type="multiclass",
            tabular_config=_tab(gen),
            cal_fraction=0.0,
            **_common(seed),
        ).fit(x, y)
        return float(f1_score(y, est.predict(x), average="macro"))

    assert _median_over_seeds(macro_f1) >= 0.60


def test_point_regressor_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)

    def r2(seed: int) -> float:
        gen = _gen("regression_point", seed)
        x, y = gen.generate(seed=seed)
        est = TFTRegressor(
            task_type="regression_point",
            tabular_config=_tab(gen),
            cal_fraction=0.0,
            **_common(seed),
        ).fit(x, y.astype(float))
        return float(r2_score(y.astype(float), est.predict(x)))

    assert _median_over_seeds(r2) >= 0.5


def test_quantile_regressor_conformal_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)

    def coverage(seed: int) -> float:
        gen = _gen("regression_point", seed)
        x, y = gen.generate(seed=seed)
        y = y.astype(float)
        est = TFTRegressor(
            task_type="regression_quantile",
            tabular_config=_tab(gen),
            quantiles=(0.1, 0.9),
            calibration_strategy="conformal",
            cal_fraction=0.2,
            **_common(seed),
        ).fit(x, y)
        q = est.predict_quantiles(x)
        lo, hi = q[:, 0], q[:, 1]
        return float(np.mean((y >= lo) & (y <= hi)))

    med = _median_over_seeds(coverage)
    assert 0.75 <= med <= 0.85
