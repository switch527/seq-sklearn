"""TFTRegressor end-to-end smoke (Phase 7, point + quantile).

Fits the real TFT regressor on a deterministic F6 synthetic regression
panel; point mode asserts predict / save-load; quantile mode asserts
the predict_quantiles contract (ordered subset, median == predict).
CPU-pinned.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.inference.attention import RegressionAttentionOutput
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[SyntheticPanelGenerator, pd.DataFrame, np.ndarray]:
    gen = SyntheticPanelGenerator(
        target_kind="regression_point",
        num_entities=20,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=8,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
    return gen, panel, y.astype(float)


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


def _estimator(gen: SyntheticPanelGenerator, **kw: object) -> TFTRegressor:
    params: dict[str, object] = {
        "task_type": "regression_point",
        "tabular_config": _tab(gen),
        "scheduler": SchedulerParams(name="constant", warmup_steps=0),
        "hidden_size": 16,
        "attention_heads": 2,
        "max_epochs": 5,
        "batch_size": 32,
        "val_fraction": 0.2,
        "cal_fraction": 0.0,
        "precision": "32-true",
        "verbose": False,
        "seed": 42,
    }
    params.update(kw)
    return TFTRegressor(**params)  # type: ignore[arg-type]


def test_tft_point_regressor_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel()
    est = _estimator(gen).fit(x, y)

    preds = est.predict(x)
    assert preds.shape == (len(x),)
    assert np.isfinite(preds).all()
    out = est.predict_with_attention(x)
    assert isinstance(out, RegressionAttentionOutput)
    assert out.quantiles_used is None
    assert np.allclose(out.predictions, est.predict(x), equal_nan=True)
    assert not hasattr(out, "logits")

    before = est.predict(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = TFTRegressor.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict(x), before)
    assert reloaded.config_ == est.config_
    assert reloaded.config_.tabular_config == est.config_.tabular_config


def test_tft_quantile_regressor_predict_quantiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel()
    est = _estimator(
        gen,
        task_type="regression_quantile",
        quantiles=(0.1, 0.5, 0.9),
    ).fit(x, y)

    assert list(est.quantiles_) == [0.1, 0.5, 0.9]
    full = est.predict_quantiles(x)
    assert full.shape == (len(x), 3)
    assert np.isfinite(full).all()
    subset = est.predict_quantiles(x, quantiles=[0.9, 0.1])
    assert subset.shape == (len(x), 2)
    assert np.array_equal(subset[:, 0], full[:, 2])
    assert np.array_equal(subset[:, 1], full[:, 0])
    # point predict == the median (0.5) column
    assert np.array_equal(est.predict(x), full[:, 1])

    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = TFTRegressor.load(tmp_path)  # type: ignore[arg-type]
    assert list(reloaded.quantiles_) == [0.1, 0.5, 0.9]
    assert np.array_equal(reloaded.predict_quantiles(x), full)
    assert reloaded.config_ == est.config_
    assert reloaded.config_.tabular_config == est.config_.tabular_config
