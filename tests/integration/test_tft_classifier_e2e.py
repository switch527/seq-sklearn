"""TFTClassifier end-to-end smoke (Phase 7, A2 / A15.1 / N1).

Fits the real TFT on a deterministic F6 synthetic binary panel through
the full TabularToSequence -> Trainer -> calibrator pipeline, then
exercises predict / predict_proba, the A17 save/load round trip
(byte-equal), and predict_with_attention's A15.1 contract. CPU-pinned
so it never contends with GPU work.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.inference.attention import AttentionOutput
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[SyntheticPanelGenerator, pd.DataFrame, np.ndarray]:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=20,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=8,
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


def _estimator(gen: SyntheticPanelGenerator, **kw: object) -> TFTClassifier:
    params: dict[str, object] = {
        "task_type": "binary",
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
    return TFTClassifier(**params)  # type: ignore[arg-type]


def test_tft_classifier_fit_predict_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel()
    est = _estimator(gen).fit(x, y)

    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = est.predict(x)
    assert preds.shape == (len(x),)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = TFTClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict_proba(x), before)
    # Pin the reconstructed TFTConfig directly: byte-equal predictions
    # can coincide even if the load()-side tabular_config re-merge
    # restored a wrong/partial config. This kills a revert of that fix.
    assert reloaded.config_ == est.config_
    assert reloaded.config_.tabular_config == est.config_.tabular_config


def test_tft_classifier_predict_with_attention_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel()
    est = _estimator(gen).fit(x, y)
    out = est.predict_with_attention(x)

    assert isinstance(out, AttentionOutput)
    n = len(x)
    length = gen.lookback
    n_vars = (
        len(gen.static_categorical_cols)
        + len(gen.static_real_cols)
        + len(gen.time_varying_real_cols)
        + len(gen.time_varying_categorical_cols)
    )
    assert out.probabilities.shape == (n, 2)
    assert out.logits.shape == (n, 1)  # binary head out_dim == 1
    assert out.predictions.shape == (n,)
    assert out.var_selection_weights.shape[0] == n
    assert out.var_selection_weights.shape[1] == length
    assert out.static_var_selection_weights.shape[0] == n
    assert out.attention_weights.shape == (
        n,
        est.config_.attention_heads,
        length,
        length,
    )
    assert out.padding_mask.shape == (n, length)
    assert out.entity_id.shape == (n,)
    # var-selection rows are softmaxes; attention rows over valid keys
    # sum to 1; nothing degenerate.
    assert np.allclose(out.var_selection_weights.sum(axis=-1), 1.0)
    assert np.allclose(out.attention_weights.sum(axis=-1), 1.0)
    assert out.var_selection_weights.std() > 0.0
    # consistency with the base predict surface (one shared forward)
    mapped = np.asarray(est.classes_)[out.predictions]
    assert np.array_equal(mapped, est.predict(x))
    assert n_vars >= 1  # the panel exercises every feature group


def _gen(**kw: object) -> SyntheticPanelGenerator:
    base: dict[str, object] = {
        "num_entities": 20,
        "periods_per_entity": 24,
        "signal_strength": 0.7,
        "lookback": 8,
        "seed": 42,
    }
    base.update(kw)
    return SyntheticPanelGenerator(**base)  # type: ignore[arg-type]


def test_tft_multiclass_predict_proba_simplex(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real TFT through the multiclass head: _head_out_dim() == num_classes
    # and the softmax branch, not exercised by the binary e2e.
    _force_cpu(monkeypatch)
    gen = _gen(target_kind="multiclass", num_classes=3)
    x, y = gen.generate(seed=42)
    est = _estimator(gen, task_type="multiclass", hidden_size=8, max_epochs=1).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(est.predict(x))).issubset(set(np.unique(y)))


def test_tft_classifier_no_categorical_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Empty categorical side: _build_tft_backbone passes [] cardinality
    # lists so the backbone's _static_pad / _tv_pad synthetic-variable
    # paths are reached through the full estimator pipeline.
    _force_cpu(monkeypatch)
    gen = _gen(num_static_categorical=0, num_time_varying_categorical=0)
    x, y = gen.generate(seed=42)
    assert not gen.static_categorical_cols
    assert not gen.time_varying_categorical_cols
    est = _estimator(gen, hidden_size=8, max_epochs=1).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
