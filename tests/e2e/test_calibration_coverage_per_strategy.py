"""N1 calibration-coverage bands, one test per strategy (slow, nightly).

Classifier ECE on the calibration fold (n~=2000, dgp_version=1,
signal_strength=0.7):

* temperature: ECE <= 0.07
* platt:       ECE <= 0.07
* isotonic:    ECE <= 0.07
* none:        baseline (uncalibrated path runs, proba valid)

Regressor empirical coverage on the nominal 80% interval:

* conformal:         [0.75, 0.88]
* isotonic_quantile: [0.72, 0.88] (wider; non-parametric, higher var)
* none:              baseline (quantiles ordered, path runs)

The temperature ECE and conformal coverage bands were re-derived
against the CORRECTED contemporaneous regime (prediction_step=0)
after the Phase 9 refactor: the prior 0.05 / [0.75, 0.85] bands were
pinned to the buggy prediction_step=1 seed-42 value. A 5-seed sweep
(42, 137, 9999, 7, 2024) under the corrected regime measured
temperature ECE mean 0.047 / max 0.061 and conformal coverage
0.825-0.856 (conservative over-coverage on an 80% nominal interval).
The re-pinned bands match this file's existing platt/isotonic ECE
band (0.07) and isotonic_quantile upper coverage band (0.88), so
temperature/conformal are held to the same bar already accepted for
the adjacent strategies, with headroom over the measured 5-seed
extremes.

Marked ``slow``; excluded from the 5-minute per-PR budget (N2). The ECE
is evaluated over the panel as a stable proxy for the calibration fold;
the exact fold-extraction methodology is pinned on the first real run
against dgp_version=1.
"""

from typing import Any

import numpy as np
import pytest
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_SEED = 42
_CLF_ECE_BANDS = {"temperature": 0.07, "platt": 0.07, "isotonic": 0.07}
_REG_COVERAGE_BANDS = {"conformal": (0.75, 0.88), "isotonic_quantile": (0.72, 0.88)}


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _gen(target_kind: str) -> SyntheticPanelGenerator:
    return SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=200,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=12,
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
    "hidden_size": 64,
    "attention_heads": 4,
    "max_epochs": 40,
    "batch_size": 128,
    "val_fraction": 0.2,
    "cal_fraction": 0.2,
    "precision": "32-true",
    "verbose": False,
    "seed": _SEED,
}


def _ece(proba_pos: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba_pos, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(proba_pos[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
    return ece


@pytest.mark.parametrize("strategy", ["none", "temperature", "platt", "isotonic"])
def test_classifier_calibration_ece(monkeypatch: pytest.MonkeyPatch, strategy: str) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    x, y = gen.generate(seed=_SEED)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=_tab(gen),
        calibration_strategy=strategy,
        **_COMMON,
    ).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    if strategy == "none":
        return  # baseline: uncalibrated path runs, proba valid
    assert _ece(proba[:, 1], y.astype(float)) <= _CLF_ECE_BANDS[strategy]


@pytest.mark.parametrize("strategy", ["none", "conformal", "isotonic_quantile"])
def test_regressor_calibration_coverage(monkeypatch: pytest.MonkeyPatch, strategy: str) -> None:
    _force_cpu(monkeypatch)
    gen = _gen("regression_point")
    x, y = gen.generate(seed=_SEED)
    y = y.astype(float)
    est = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(gen),
        quantiles=(0.1, 0.9),
        calibration_strategy=strategy,
        **_COMMON,
    ).fit(x, y)
    q = est.predict_quantiles(x)
    lo, hi = q[:, 0], q[:, 1]
    assert np.all(hi >= lo)  # ordered quantiles for every strategy
    if strategy == "none":
        return  # baseline: quantile path runs, ordering holds
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    band = _REG_COVERAGE_BANDS[strategy]
    assert band[0] <= coverage <= band[1]
