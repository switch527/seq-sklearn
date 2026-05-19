"""End-to-end estimator-shell smoke (Phase 6a, N1).

Fits the synthetic ``_DummySequenceClassifier`` through the full
TabularToSequence -> Trainer -> calibrator pipeline, then exercises
predict_proba and the A17 save / load round trip (same process,
byte-equal predictions). A second variant runs the temperature
calibrator so serialize / deserialize is exercised through the
integrated save / load path.
"""

import numpy as np
import pandas as pd
import pytest
import sklearn.base
import torch

from seq_sklearn.config.adapters import (
    LossParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
)
from seq_sklearn.errors import ConfigError
from tests._test_models._dummy_estimator import _DummySequenceClassifier


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _binary_panel(n_entities: int = 24, n_steps: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(n_entities):
        label = e % 2
        sr = float(label) + rng.normal(0, 0.1)
        for t in range(n_steps):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "sr": sr,
                    "tr": float(label) + 0.3 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def _tab() -> TabularConfigParams:
    return TabularConfigParams(
        static_real_cols=("sr",),
        time_varying_real_cols=("tr",),
        lookback=4,
        min_periods=1,
        min_periods_predict=1,
    )


def _estimator(**kw: object) -> _DummySequenceClassifier:
    params: dict[str, object] = {
        "task_type": "binary",
        "tabular_config": _tab(),
        "scheduler": SchedulerParams(name="constant", warmup_steps=0),
        "max_epochs": 2,
        "batch_size": 16,
        "val_fraction": 0.2,
        "cal_fraction": 0.0,
        "precision": "32-true",
        "verbose": False,
        "seed": 42,
    }
    params.update(kw)
    return _DummySequenceClassifier(**params)  # type: ignore[arg-type]


def test_dummy_classifier_fit_predict_proba(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    est = _estimator().fit(x, y)
    proba = est.predict_proba(x)
    assert proba.ndim == 2
    assert proba.shape[1] == 2
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert est.n_features_in_ == 2
    assert list(est.feature_names_in_) == ["sr", "tr"]


def test_save_load_same_process_byte_equal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    est = _estimator().fit(x, y)
    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceClassifier.load(tmp_path)  # type: ignore[arg-type]
    after = reloaded.predict_proba(x)
    assert np.array_equal(after, before)
    # A2: binary w/o threshold_tuning -> attribute ABSENT after load,
    # not silently resurrected from a stray state.json key.
    assert not hasattr(reloaded, "decision_threshold_")


def test_save_load_with_temperature_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    # cal_fraction > 0 so the three-way split carves a calibration fold
    # the temperature calibrator fits on.
    est = _estimator(calibration_strategy="temperature", cal_fraction=0.2).fit(x, y)
    assert est.calibrator_ is not None
    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict_proba(x), before)


def _multiclass_panel(n_entities: int = 24, n_steps: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(n_entities):
        label = e % 3
        for t in range(n_steps):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "sr": float(label) + rng.normal(0, 0.1),
                    "tr": float(label) + 0.3 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def test_multiclass_isotonic_predict_and_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _multiclass_panel()
    est = _DummySequenceClassifier(
        task_type="multiclass",
        tabular_config=_tab(),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        max_epochs=2,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.2,
        calibration_strategy="isotonic",
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape[1] == 3
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = est.predict(x)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert not hasattr(est, "decision_threshold_")  # A2: multiclass absent
    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict_proba(x), before)
    assert not hasattr(reloaded, "decision_threshold_")  # A2 survives load


def test_binary_platt_with_threshold_tuning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    est = _estimator(
        calibration_strategy="platt",
        cal_fraction=0.2,
        threshold_tuning=True,
        threshold_metric="f1",
    ).fit(x, y)
    assert hasattr(est, "decision_threshold_")
    assert 0.0 <= est.decision_threshold_ <= 1.0
    before = est.predict(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert reloaded.decision_threshold_ == est.decision_threshold_
    assert np.array_equal(reloaded.predict(x), before)


def test_binary_threshold_tuning_without_calibrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # threshold_tuning on but calibration_strategy='none': the tuner
    # fits on sigmoid(raw) (no calibrator), exercising that branch.
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    est = _estimator(threshold_tuning=True, cal_fraction=0.2).fit(x, y)
    assert est.calibrator_ is None
    assert hasattr(est, "decision_threshold_")
    assert 0.0 <= est.decision_threshold_ <= 1.0


def test_explicit_calibration_set_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    x_cal, y_cal = _binary_panel(n_entities=8)
    est = _estimator(calibration_strategy="temperature", cal_fraction=0.0).fit(
        x, y, calibration_set=(x_cal, y_cal)
    )
    assert est.calibrator_ is not None
    proba = est.predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_load_with_explicit_loss_adapter_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # Gemini integration CRITICAL: __init__ stores `loss` verbatim as
    # None (the one adapter not defaulted), so when an explicit loss
    # adapter is saved, _hyperparams persists `loss__*` leaves but drops
    # the bare object; load()'s set_params then routed `loss__strategy`
    # to None.set_params -> AttributeError. Every other test uses the
    # loss=None default, so this shipped untested. The reloaded
    # estimator must both predict byte-equal and survive clone().
    _force_cpu(monkeypatch)
    x, y = _binary_panel()
    est = _estimator(
        loss=LossParams(strategy="cross_entropy", label_smoothing=0.1, focal_gamma=3.0)
    ).fit(x, y)
    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict_proba(x), before)
    # every loss leaf (not just strategy) survives restore -> set_params
    params = reloaded.get_params(deep=True)
    assert params["loss__strategy"] == "cross_entropy"
    assert params["loss__label_smoothing"] == 0.1
    assert params["loss__focal_gamma"] == 3.0
    clone = sklearn.base.clone(reloaded)  # must not raise
    assert clone.get_params(deep=True)["loss__label_smoothing"] == 0.1


def _mixed_multiclass_panel() -> tuple[pd.DataFrame, np.ndarray]:
    """16 above-floor (8-row) + 4 below-floor (2-row) entities, 3 classes."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    spec = [(e, 8) for e in range(16)] + [(100 + e, 2) for e in range(4)]
    for eid, n in spec:
        label = eid % 3
        sr = float(label) + rng.normal(0, 0.1)
        for t in range(n):
            rows.append(
                {
                    "id": eid,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "sr": sr,
                    "tr": float(label) + 0.3 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def _below_tab() -> TabularConfigParams:
    return TabularConfigParams(
        static_real_cols=("sr",),
        time_varying_real_cols=("tr",),
        lookback=4,
        min_periods=1,
        min_periods_predict=3,
    )


def test_fit_filters_below_floor_classifier_class_weighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other C2 arm: classifier multiclass + class_weighted reaches
    # torch.bincount over the train fold; an unfiltered -1 sentinel
    # (below-floor classification target) raises RuntimeError. The
    # regressor regression test only covers the NaN->F9 arm, so this
    # pins the bincount arm independently.
    _force_cpu(monkeypatch)
    x, y = _mixed_multiclass_panel()
    est = _estimator(
        task_type="multiclass",
        tabular_config=_below_tab(),
        sampler=SamplerParams(strategy="class_weighted"),
        loss=LossParams(strategy="cross_entropy"),
    ).fit(x, y)  # must not raise RuntimeError from bincount(-1)
    proba = est.predict_proba(x)
    assert proba.shape[0] == len(x)
    assert np.allclose(np.nansum(proba, axis=1)[: 16 * 8], 1.0)


def test_fit_all_below_floor_raises_configerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Empty-fold guard symmetric with the estimator's
    # _calibration_fold: every entity below min_periods_predict means
    # train/val empty after the filter -> loud ConfigError, not a
    # silent empty Lightning loader.
    _force_cpu(monkeypatch)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(12):  # every entity 2 rows, floor is 3
        for t in range(2):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "sr": 0.0,
                    "tr": float(rng.normal()),
                }
            )
            y.append(e % 2)
    est = _estimator(task_type="binary", tabular_config=_below_tab())
    with pytest.raises(ConfigError, match="empty after dropping below-floor"):
        est.fit(pd.DataFrame(rows), np.asarray(y))
