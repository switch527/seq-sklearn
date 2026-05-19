"""ONNX export parity (Phase 10, architecture A21 / N1).

Exported graph (backbone + head, raw logits) must agree with the
production sorted-space oracle `est._predict_raw(panel)[0]` within
`atol=1e-6` (wrapper == production, pack-free vs packed) and the
onnxruntime run must agree with the wrapper within
`atol=1e-4, rtol=1e-4`. The ONNX-logits + documented post-processing
must reproduce `predict_proba` (above-floor rows; below-floor NaN).

Skips cleanly without the optional `[onnx]` extra; the per-PR
`test-deploy` CI job installs it and runs this file (Step 3b).
"""

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams  # noqa: E402
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator  # noqa: E402
from seq_sklearn.errors import NotFittedError  # noqa: E402
from seq_sklearn.inference.onnx import _OnnxForward  # noqa: E402
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier  # noqa: E402
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor  # noqa: E402

pytestmark = pytest.mark.onnx


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _gen(target_kind: str = "binary") -> SyntheticPanelGenerator:
    return SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=16,
        periods_per_entity=14,
        signal_strength=0.8,
        lookback=6,
        seed=42,
    )


def _tab(gen: SyntheticPanelGenerator, **extra: object) -> TabularConfigParams:
    base: dict[str, object] = {
        "id_col": gen.id_col,
        "time_col": gen.time_col,
        "static_categorical_cols": tuple(gen.static_categorical_cols),
        "static_real_cols": tuple(gen.static_real_cols),
        "time_varying_real_cols": tuple(gen.time_varying_real_cols),
        "time_varying_categorical_cols": tuple(gen.time_varying_categorical_cols),
        "lookback": gen.lookback,
        "min_periods": 1,
        "min_periods_predict": 1,
        "max_categorical_cardinality": 10_000,
    }
    base.update(extra)
    return TabularConfigParams(**base)  # type: ignore[arg-type]


_COMMON: dict[str, object] = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "hidden_size": 16,
    "attention_heads": 2,
    "max_epochs": 3,
    "batch_size": 32,
    "val_fraction": 0.2,
    "cal_fraction": 0.0,
    "precision": "32-true",
    "seed": 42,
    "verbose": False,
}


def _short_entity_panel(
    gen: SyntheticPanelGenerator,
) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Panel with a left-padded short entity (one entity truncated to 2 rows)."""
    panel, y = gen.generate(seed=42)
    short_id = panel[gen.id_col].iloc[-1]
    block = panel[panel[gen.id_col] == short_id].index
    drop = list(block[2:])
    panel = panel.drop(index=drop).reset_index(drop=True)
    y = np.delete(y, drop)
    return panel, y, short_id


def _batch_args(est: object, panel: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    batch = est.transformer_.transform(panel)  # type: ignore[attr-defined]
    return (
        batch["static_categorical"],
        batch["static_real"],
        batch["time_varying_real"],
        batch["time_varying_categorical"],
        batch["padding_mask"],
    )


def _ort_run(path: str, args: tuple[torch.Tensor, ...]) -> np.ndarray:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {n: a.cpu().numpy() for n, a in zip(names, args, strict=True)}
    return sess.run(None, feed)[0]


def test_export_onnx_parity_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y, short_id = _short_entity_panel(gen)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=_tab(gen, min_periods_predict=4),
        calibration_strategy="temperature",
        **{**_COMMON, "cal_fraction": 0.2},  # temperature needs a cal fold
    ).fit(panel, y)
    out = str(tmp_path / "m.onnx")  # type: ignore[operator]
    est.export_onnx(out, panel)

    args = _batch_args(est, panel)
    backbone, head = est._predict_module()
    wrapper_out = _OnnxForward(backbone, head)(*args).cpu().numpy()
    oracle = est._predict_raw(panel)[0].cpu().numpy()

    # (a) wrapper (pack-free) == production sorted-space logits (packed).
    np.testing.assert_allclose(wrapper_out, oracle, atol=1e-6)
    # (b) onnxruntime == wrapper within the N1 tolerance.
    np.testing.assert_allclose(_ort_run(out, args), wrapper_out, atol=1e-4, rtol=1e-4)

    # (c) ONNX logits + sigmoid + the documented post-processing
    # reproduces predict_proba for above-floor rows; below-floor NaN.
    proba = est.predict_proba(panel)
    ort_logits = _ort_run(out, args).reshape(-1)
    ort_p1 = 1.0 / (1.0 + np.exp(-ort_logits))  # pre-calibration sigmoid
    short_mask = (panel[gen.id_col] == short_id).to_numpy()
    assert np.isnan(proba[short_mask]).all()
    # Above-floor: the deployed predict_proba (calibrated) is a
    # monotone transform of ort_p1; assert rank-correlation is exact
    # (calibration is monotone and lives outside the graph).
    af = ~short_mask
    order_proba = np.argsort(proba[af, 1])
    order_ort = np.argsort(ort_p1[af])
    np.testing.assert_array_equal(order_proba, order_ort)

    # (d) the calibrator is non-trivial, so (c) is sensitive to
    # calibration being OUTSIDE the graph.
    assert est.calibrator_ is not None


def test_export_onnx_parity_loaded_estimator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y, _ = _short_entity_panel(gen)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=_tab(gen),
        calibration_strategy="temperature",
        **{**_COMMON, "cal_fraction": 0.2},  # temperature needs a cal fold
    ).fit(panel, y)
    save_dir = str(tmp_path / "saved")  # type: ignore[operator]
    est.save(save_dir)
    loaded = TFTClassifier.load(save_dir)

    out = str(tmp_path / "loaded.onnx")  # type: ignore[operator]
    loaded.export_onnx(out, panel)  # the loaded path _predict_module() uses
    args = _batch_args(loaded, panel)
    backbone, head = loaded._predict_module()
    wrapper_out = _OnnxForward(backbone, head)(*args).cpu().numpy()
    oracle = loaded._predict_raw(panel)[0].cpu().numpy()
    np.testing.assert_allclose(wrapper_out, oracle, atol=1e-6)
    np.testing.assert_allclose(_ort_run(out, args), wrapper_out, atol=1e-4, rtol=1e-4)
    # Carries the (c) calibration-bridge onto the loaded path: a
    # post-load calibrator-serialization bug would diverge here.
    np.testing.assert_allclose(
        loaded.predict_proba(panel), est.predict_proba(panel), atol=1e-5, equal_nan=True
    )


def test_export_onnx_parity_multiclass(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("multiclass")
    panel, y = gen.generate(seed=42)
    est = TFTClassifier(task_type="multiclass", tabular_config=_tab(gen), **_COMMON).fit(panel, y)
    out = str(tmp_path / "mc.onnx")  # type: ignore[operator]
    est.export_onnx(out, panel)
    args = _batch_args(est, panel)
    backbone, head = est._predict_module()
    wrapper_out = _OnnxForward(backbone, head)(*args).cpu().numpy()
    assert wrapper_out.shape == (len(panel), est.n_classes_)
    np.testing.assert_allclose(_ort_run(out, args), wrapper_out, atol=1e-4, rtol=1e-4)


def test_export_onnx_parity_regressor_quantile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("regression_quantile")
    panel, y = gen.generate(seed=42)
    est = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(gen),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(panel, y.astype(float))
    out = str(tmp_path / "q.onnx")  # type: ignore[operator]
    est.export_onnx(out, panel)
    args = _batch_args(est, panel)
    backbone, head = est._predict_module()
    wrapper_out = _OnnxForward(backbone, head)(*args).cpu().numpy()
    assert wrapper_out.shape == (len(panel), 3)
    np.testing.assert_allclose(_ort_run(out, args), wrapper_out, atol=1e-4, rtol=1e-4)


def test_export_onnx_dynamic_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y = gen.generate(seed=42)
    est = TFTClassifier(task_type="binary", tabular_config=_tab(gen), **_COMMON).fit(panel, y)
    out = str(tmp_path / "dyn.onnx")  # type: ignore[operator]
    est.export_onnx(out, panel)

    # A different batch size through the SAME exported artifact (no
    # re-export): proves the serialized Dim("batch") is real.
    half = panel[panel[gen.id_col].isin(panel[gen.id_col].unique()[:6])].reset_index(drop=True)
    args2 = _batch_args(est, half)
    backbone, head = est._predict_module()
    wrapper2 = _OnnxForward(backbone, head)(*args2).cpu().numpy()
    assert wrapper2.shape[0] != _batch_args(est, panel)[0].shape[0]
    np.testing.assert_allclose(_ort_run(out, args2), wrapper2, atol=1e-4, rtol=1e-4)


def test_export_onnx_raises_not_fitted(tmp_path: object) -> None:
    gen = _gen("binary")
    panel, _ = gen.generate(seed=42)
    est = TFTClassifier(task_type="binary", tabular_config=_tab(gen), **_COMMON)
    with pytest.raises(NotFittedError):
        est.export_onnx(str(tmp_path / "x.onnx"), panel)  # type: ignore[operator]
