"""TFT output snapshot regression, one per task type (slow, nightly).

Pins the deterministic CPU FP32 prediction for each v1 task type
against an artifact in ``tests/_snapshots/`` plus a manifest
(dgp_version, seed, config hash). A drift in the model output, the DGP,
or the config trips the snapshot; the artifacts are refreshed only via
``pytest tests/snapshot/ --snapshot-update`` and a commit carrying the
``SNAPSHOT_REVIEWED:`` marker (A14 / N1).

Marked ``slow``: a TFT forward pass at this config exceeds the 2s
threshold, so it runs nightly and is excluded from the per-PR budget.
"""

import hashlib
import json
from typing import Any

import numpy as np
import pytest
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

pytestmark = pytest.mark.slow

_SEED = 42


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _gen(target_kind: str, num_classes: int = 3) -> SyntheticPanelGenerator:
    return SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=24,
        periods_per_entity=18,
        signal_strength=0.7,
        lookback=8,
        num_classes=num_classes,
        seed=_SEED,
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


_COMMON: dict[str, Any] = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "hidden_size": 16,
    "attention_heads": 2,
    "max_epochs": 3,
    "batch_size": 32,
    "val_fraction": 0.2,
    "cal_fraction": 0.0,
    "precision": "32-true",
    "verbose": False,
    "seed": _SEED,
}


def _manifest(gen: SyntheticPanelGenerator, est: object) -> dict[str, object]:
    config_json = json.dumps(
        est.config_.model_dump(mode="json"),  # type: ignore[attr-defined]
        sort_keys=True,
    )
    return {
        "dgp_version": gen.dgp_version,
        "seed": _SEED,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest()[:16],
    }


def test_binary_classifier_snapshot(monkeypatch: pytest.MonkeyPatch, snapshot: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    x, y = gen.generate(seed=_SEED)
    est = TFTClassifier(task_type="binary", tabular_config=_tab(gen), **_COMMON).fit(x, y)
    out = est.predict_proba(x).astype(np.float64)
    snapshot.check("tft_binary_proba", out, _manifest(gen, est))  # type: ignore[attr-defined]


def test_multiclass_classifier_snapshot(monkeypatch: pytest.MonkeyPatch, snapshot: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("multiclass", num_classes=3)
    x, y = gen.generate(seed=_SEED)
    est = TFTClassifier(task_type="multiclass", tabular_config=_tab(gen), **_COMMON).fit(x, y)
    out = est.predict_proba(x).astype(np.float64)
    snapshot.check("tft_multiclass_proba", out, _manifest(gen, est))  # type: ignore[attr-defined]


def test_point_regressor_snapshot(monkeypatch: pytest.MonkeyPatch, snapshot: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("regression_point")
    x, y = gen.generate(seed=_SEED)
    est = TFTRegressor(task_type="regression_point", tabular_config=_tab(gen), **_COMMON).fit(
        x, y.astype(float)
    )
    out = est.predict(x).astype(np.float64)
    snapshot.check("tft_regression_point", out, _manifest(gen, est))  # type: ignore[attr-defined]


def test_quantile_regressor_snapshot(monkeypatch: pytest.MonkeyPatch, snapshot: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("regression_point")
    x, y = gen.generate(seed=_SEED)
    est = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(gen),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(x, y.astype(float))
    out = est.predict_quantiles(x).astype(np.float64)
    snapshot.check("tft_regression_quantile", out, _manifest(gen, est))  # type: ignore[attr-defined]
