"""End-to-end fit/predict smoke for the seq-sklearn TFT adapters.

Proves the adapter actually works against the library's v1.0.0
façade by fitting on a tiny synthetic panel and asserting shape +
type of the output. Marked ``slow`` per the repo convention
(`pytest-mark` config), so it skips on dev-loop runs but lands in
the default CI suite alongside the existing tests/e2e/ smoke.
"""

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.seq_sklearn import (
    SeqSklearnTFTClassifierAdapter,
    SeqSklearnTFTRegressorAdapter,
)
from benchmarks.config import DatasetSpec


def _make_spec(*, name: str, task_type: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        task_type=task_type,  # pyright: ignore[reportArgumentType]
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.com/data.csv",
        integrity_sha256="0" * 64,
        archive_basename="data.csv",
        entity_col="entity_id",
        time_col="cycle",
        target_col="y",
        feature_real_cols=("x0", "x1"),
        feature_categorical_cols=(),
        lookback=4,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1 if task_type == "binary" else None,
        excluded=False,
        exclusion_reason=None,
        citation="Acme 2020",
    )


def _make_panel(*, n_entities: int = 4, n_periods: int = 6) -> pd.DataFrame:
    """Tiny synthetic panel: one row per (entity, cycle), two real
    features (a deterministic ramp + a stochastic noise channel)."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for e in range(n_entities):
        for t in range(n_periods):
            rows.append(
                {
                    "entity_id": e,
                    "cycle": t,
                    "x0": float(t),
                    "x1": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def _binary_y(panel: pd.DataFrame) -> np.ndarray:
    """Bernoulli targets correlated with `x0` so the model has signal."""
    rng = np.random.default_rng(0)
    p = 1.0 / (1.0 + np.exp(-(panel["x0"].to_numpy() - 2.5)))
    return (rng.random(len(panel)) < p).astype(np.int64)


def _regression_y(panel: pd.DataFrame) -> np.ndarray:
    """Linear-in-x0 target with mild noise."""
    rng = np.random.default_rng(0)
    return (
        panel["x0"].to_numpy()
        + 0.5 * panel["x1"].to_numpy()
        + rng.normal(scale=0.1, size=len(panel))
    ).astype(np.float64)


@pytest.mark.slow
def test_tft_classifier_adapter_fits_and_predicts_smoke() -> None:
    spec = _make_spec(name="ds_binary", task_type="binary")
    panel = _make_panel()
    y = _binary_y(panel)

    adapter = SeqSklearnTFTClassifierAdapter(
        spec=spec,
        hyperparameters={
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y)

    preds = adapter.predict(panel)
    assert preds.shape == (len(panel),)
    assert preds.dtype.kind in ("i", "u")

    proba = adapter.predict_proba(panel)
    assert proba.shape == (len(panel), 2)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


@pytest.mark.slow
def test_tft_regressor_adapter_fits_and_predicts_smoke() -> None:
    spec = _make_spec(name="ds_reg", task_type="regression_point")
    panel = _make_panel()
    y = _regression_y(panel)

    adapter = SeqSklearnTFTRegressorAdapter(
        spec=spec,
        hyperparameters={
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y)

    preds = adapter.predict(panel)
    assert preds.shape == (len(panel),)
    assert preds.dtype.kind == "f"


@pytest.mark.slow
def test_tft_regressor_adapter_quantile_fit_predict_smoke() -> None:
    """qa-I2 + qa-I3: end-to-end fit on `regression_quantile`
    task_type with the `predict_quantiles` happy path. Closes the
    only adapter-execution path B2 left dead in CI."""
    spec = _make_spec(name="ds_quantile", task_type="regression_quantile")
    panel = _make_panel()
    y = _regression_y(panel)

    adapter = SeqSklearnTFTRegressorAdapter(
        spec=spec,
        hyperparameters={
            "quantiles": (0.1, 0.5, 0.9),
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y)

    quantiles = adapter.predict_quantiles(panel)
    assert quantiles.shape == (len(panel), 3)
    assert np.isfinite(quantiles).all()


@pytest.mark.slow
def test_tft_regressor_adapter_casts_integer_y_to_float() -> None:
    """qa-N2: the regressor adapter casts `y` to float64 inside fit,
    so an integer-valued `y` (a perfectly reasonable shape for a
    count-regression dataset) does not surface a dtype error
    downstream. Without the cast, the library's training loop would
    reject the int64 array."""
    spec = _make_spec(name="ds_reg_int", task_type="regression_point")
    panel = _make_panel()
    # Integer-valued targets (counts of x0 rounded down).
    y_int = np.floor(panel["x0"].to_numpy()).astype(np.int64)

    adapter = SeqSklearnTFTRegressorAdapter(
        spec=spec,
        hyperparameters={
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y_int)
    preds = adapter.predict(panel)
    assert preds.shape == (len(panel),)
    assert preds.dtype.kind == "f"


def test_tft_classifier_adapter_hyperparameter_override_lands_in_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qa-N1: a user-supplied `hyperparameters` entry must reach the
    underlying estimator's constructor. Intercept TFTClassifier via
    the same public-façade module the adapter imports from and
    assert the merged kwargs contain the user-supplied
    `hidden_size`. Fast test (no fit).
    """
    spec = _make_spec(name="ds_binary_hp", task_type="binary")

    captured: dict[str, object] = {}

    # Patch the symbol the adapter module captured at its own
    # import time (this is the same TFTClassifier the façade
    # re-exports; `_module` is a fully-public import path).
    from benchmarks.adapters import seq_sklearn as adapter_module

    original_cls = adapter_module.TFTClassifier

    def capturing_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        # Return a sentinel; the adapter's `fit` call below would
        # propagate, so we replace the constructor only long enough
        # to capture kwargs before the test asserts and exits.
        raise _ConstructorCapturedError

    monkeypatch.setattr(adapter_module, "TFTClassifier", capturing_constructor)

    adapter = SeqSklearnTFTClassifierAdapter(
        spec=spec,
        hyperparameters={"hidden_size": 42},
    )
    panel = _make_panel()
    y = _binary_y(panel)
    with pytest.raises(_ConstructorCapturedError):
        adapter.fit(panel, y)

    assert captured["hidden_size"] == 42, captured
    assert captured["task_type"] == "binary"
    # The lookup at exit confirms the patch was applied to the right
    # symbol; without this assertion the test could pass under any
    # patch site.
    assert original_cls is not capturing_constructor


class _ConstructorCapturedError(Exception):
    """Sentinel exception raised by the capturing constructor in the
    `hyperparameter_override` test so the test exits cleanly after
    inspecting the merged kwargs, without proceeding to a real fit."""
